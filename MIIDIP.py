import argparse
import cv2
import glob
import os
import torch

from basicsr.utils import img2tensor
from basicsr.archs.dehaze_vq_weight_DQA_arch import VQWeightDehazeNet_DQA


def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='./examples/0047',
                        help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str,
                        default='./pretrained_models/released.pth',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--use_weight', default='./pretrained_models/weight_for_matching_dehazing_Flickr.pth')
    parser.add_argument('--alpha', type=float, default=21.25, help='value of alpha')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=1500, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weight_path = args.weight

    # set up the model
    sr_model = VQWeightDehazeNet_DQA(codebook_params=[[64, 1024, 512]], LQ_stage=True, use_weight=args.use_weight, weight_alpha=args.alpha).to(device)
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    sr_model.eval()

    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # path_gt = path_gt + img_name.split('_')[1]+'_0.jpg'
        path_gt = './examples/gt.jpg'
        img_gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
        if img.max() > 255.0:
            img = img / 255.0
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        img_tensor = img2tensor(img).to(device) / 255.
        img_gt_tensor = img2tensor(img_gt).to(device) / 255.

        img_tensor = img_tensor.unsqueeze(0)
        img_gt_tensor = img_gt_tensor.unsqueeze(0)
        print(img_name)
        with torch.no_grad():
            output, GT_index, DEHAZE_index = sr_model.test_DQA(img_tensor, img_gt_tensor)
            print(output.round(4))


if __name__ == '__main__':
    main()
