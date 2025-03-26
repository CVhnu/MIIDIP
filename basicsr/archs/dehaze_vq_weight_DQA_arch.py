import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv

from .network_swinir import RSTB
from .MIIDIP_utils import ResBlock, CombineQuantBlock



class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                     self.dilation, self.groups, self.deformable_groups)


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=1, LQ_stage=False, weight_alpha=1.0):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta
        self.weight_alpha = weight_alpha
        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def dist(self, x, y):
        if x.shape == y.shape:
            return (x - y) ** 2
        else:
            return torch.sum(x ** 2, dim=1, keepdim=True) + \
                torch.sum(y ** 2, dim=1) - 2 * \
                torch.matmul(x, y.t())

    def cosine(self, x, y):
        if x.shape == y.shape:
            return F.cosine_similarity(x,y,dim=1)


    def forward(self, z, gt):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        gt = gt.permute(0, 2, 3, 1).contiguous()
        gt_flattened = gt.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        d_gt = self.dist(gt_flattened, codebook)

        # find closest encodings
        min_encoding_indices_gt = torch.argmin(d_gt, dim=1).unsqueeze(1)
        min_encodings_gt = torch.zeros(min_encoding_indices_gt.shape[0], codebook.shape[0]).to(gt)
        min_encodings_gt.scatter_(1, min_encoding_indices_gt, 1)


        GT = min_encoding_indices_gt.cpu().numpy()
        DEHAZE = min_encoding_indices.cpu().numpy()

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        z_q_gt = torch.matmul(min_encodings_gt, codebook)
        z_q_gt = z_q_gt.view(z.shape)

        ##################################
        cosine_score = self.cosine(z_q_gt.detach(), z_q.detach())
        cosine_score = cosine_score.cpu().numpy().round(4)
        return cosine_score, GT, DEHAZE

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                 blk_depth=6,
                 num_heads=8,
                 window_size=8,
                 **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()
        self.LQ_stage = LQ_stage
        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage:
            self.blocks.append(SwinLayers(**swin_opts))

    def forward(self, input):
        x = self.in_conv(input)
        for idx, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


class WarpBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.offset = nn.Conv2d(in_channel * 2, in_channel, 3, stride=1, padding=1)
        self.dcn = DCNv2Pack(in_channel, in_channel, 3, padding=1, deformable_groups=4)

    def forward(self, x_vq, x_residual):
        x_residual = self.offset(torch.cat([x_vq, x_residual], dim=1))
        feat_after_warp = self.dcn(x_vq, x_residual)

        return feat_after_warp


class MultiScaleDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 only_residual=False,
                 use_warp=True
                 ):
        super().__init__()
        self.only_residual = only_residual
        self.use_warp = use_warp
        self.upsampler = nn.ModuleList()
        self.warp = nn.ModuleList()
        res = input_res // (2 ** max_depth)
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            self.upsampler.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                ResBlock(out_channel, out_channel, norm_type, act_type),
            )
            )
            self.warp.append(WarpBlock(out_channel))
            res = res * 2

    def forward(self, input, code_decoder_output):
        x = input
        for idx, m in enumerate(self.upsampler):
            with torch.backends.cudnn.flags(enabled=False):
                if not self.only_residual:
                    x = m(x)
                    if self.use_warp:
                        x_vq = self.warp[idx](code_decoder_output[idx], x)
                        x = x + x_vq * (x.mean() / x_vq.mean())
                    else:
                        x = x + code_decoder_output[idx]
                else:
                    x = m(x)

        return x


@ARCH_REGISTRY.register()
class VQWeightDehazeNet_DQA(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_residual=True,
                 only_residual=False,
                 use_warp=True,
                 weight_alpha=1.0,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.use_residual = use_residual
        self.only_residual = only_residual
        self.use_warp = use_warp
        self.weight_alpha = weight_alpha

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            self.max_depth,
            self.gt_res,
            channel_query_dict,
            norm_type, act_type, LQ_stage
        )
        if self.LQ_stage and self.use_residual:
            self.multiscale_decoder = MultiScaleDecoder(
                in_channel,
                self.max_depth,
                self.gt_res,
                channel_query_dict,
                norm_type, act_type, only_residual, use_warp=self.use_warp
            )

        # build decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
                weight_alpha=self.weight_alpha
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim[scale]
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))


    def encode_and_decode(self, input, gt):
        enc_feats = self.multiscale_encoder(input)
        enc_feats_gt = self.multiscale_encoder(gt)

        quant_idx = 0
        x = enc_feats
        x_gt = enc_feats_gt

        feat_to_quant = self.before_quant_group[quant_idx](x)
        feat_to_quant_gt = self.before_quant_group[quant_idx](x_gt)

        with torch.no_grad():
            codebook_score = self.quantize_group[quant_idx](feat_to_quant, feat_to_quant_gt)

        return codebook_score

    @torch.no_grad()
    def test_DQA(self, input, gt_input, weight_alpha=None):

        wsz = 32
        _, _, h_old, w_old = input.shape
        with torch.no_grad():
            if (h_old % wsz) != 0 or (w_old % wsz) != 0:
                # padding to multiple of window_size * 8

                h_pad = (h_old // wsz + 1) * wsz - h_old
                w_pad = (w_old // wsz + 1) * wsz - w_old
                input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
                input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

                _, _, h_old, w_old = gt_input.shape
                h_pad = (h_old // wsz + 1) * wsz - h_old
                w_pad = (w_old // wsz + 1) * wsz - w_old
                gt_input = torch.cat([gt_input, torch.flip(gt_input, [2])], 2)[:, :, :h_old + h_pad, :]
                gt_input = torch.cat([gt_input, torch.flip(gt_input, [3])], 3)[:, :, :, :w_old + w_pad]

            codebook_socre, GT_index, DEHAZE_index = self.encode_and_decode(input, gt_input)
            DQA = codebook_socre.mean()


        return DQA, GT_index, DEHAZE_index