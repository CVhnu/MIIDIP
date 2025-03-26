# MIIDIP
Shift-Tolerant Perceptual Similarity Metric for Full Reference Image Dehazing Quality Assessment

# Dependencies and Installation
Ubuntu >= 18.04  
CUDA >= 11.0  
Python 3.6  
Pytorch 1.2.0  

\# git clone this repository
git clone https://github.com/CVhnu/MIIDIP.git  
cd MIIDIP

\# create new anaconda env
conda create -n SSID python=3.8  
conda activate MIIDIP  

# Get Started
1. The pretrained checkpoints in the Code/pretrained/MIIDIP.pt.
2. Preparing data for training

# Quick test
Run demos to process the images in dir ./examples/examples/ by following commands:  
python MIIDIP.py 

# The test results on the three datasets(RW-HAZE, BeDDE, MRFID).
<img src=https://github.com/CVhnu/MIIDIP/blob/main/test.png >

## The released model can be downloaded at
* [Baidu](https://pan.baidu.com/s/1-zOBkKkAu9yHX1JgXAEgzw)[password=vh78]
* [Google](https://drive.google.com/file/d/1zJqhScPxTvPlq5WhBRg29d-W_lfsp-4I/view?usp=drive_link)


# Citation
If you find our repo useful for your research, please cite us:


# License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.

# Acknowledgement

Code is inspired by ([RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing)) and ([BasicSR](https://github.com/XPixelGroup/BasicSR)) .
