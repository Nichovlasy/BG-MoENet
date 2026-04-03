# BG-MoENet: A Boundary-Guided Mixture-of-Experts Network for Power Line Segmentation in UAV Images

## Network Architecture

![BG-MoE Net](asset\BG-MoE Net.jpg)

## Quantitative Performance

| Models                                                       | Params(M) | FLOPs(G) | F1(%) | IoU(%) | Precision(%) | Recall(%) |
| ------------------------------------------------------------ | --------- | -------- | ----- | ------ | ------------ | --------- |
| [UNet](https://arxiv.org/pdf/1505.04597)                     | 31.04     | 218.97   | 52.95 | 69.24  | 75.76        | 63.75     |
| [SegFormer-B0](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf) | 3.71      | 6.76     | 55.91 | 71.72  | 73.94        | 69.63     |
| [SegNeXt-T](https://proceedings.neurips.cc/paper_files/paper/2022/file/08050f40fff41616ccfc3080e60a301a-Paper-Conference.pdf) | 4.30      | 6.60     | 51.07 | 67.61  | 74.09        | 62.17     |
| [DeepLab V3+](https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf) | 41.22     | 176.71   | 52.80 | 69.11  | 77.84        | 62.14     |
| [DDRNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_DetectoRS_Detecting_Objects_With_Recursive_Feature_Pyramid_and_Switchable_Atrous_CVPR_2021_paper.pdf) | 20.14     | 17.97    | 43.95 | 61.06  | 72.97        | 52.50     |
| [Mask2Former-Swin-T](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf) | 47.40     | 27.57    | 58.67 | 73.95  | 70.37        | 77.93     |
| [PLNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10816189) | 40.64     | 90.76    | 58.41 | 73.74  | 69.48        | 78.57     |
| [MiT-Unet-B0](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10877786) | 4.87      | 18.48    | 55.40 | 71.30  | 77.22        | 66.22     |
| BG-MoENet-B0 (Ours)                                          | 10.53     | 17.07    | 60.26 | 75.20  | 75.15        | 75.25     |

## Results

![Results](asset\Results.jpg)

## Require 

Please `pip install` the following packages:

- Python>=3.10
- torch>=2.3.1+cu121
- torchaudio>=2.3.1+cu121
- torchvision>=0.18.1+cu121
- mmcv>=2.2.0
- mmsegmentation>=1.2.2
- mmengine>=0.10.7
- matplotlib>=3.10.6
- numpy>=2.1.2
- opencv>=4.12.0
- scipy>=1.15.3
- ftfy>=6.3.1
- regex
- fvcore

## Test:

1.`git clone https://github.com/open-mmlab/mmsegmentation.git`

or Visit the official Github of mmsegmentation for cloning. ([Link](https://github.com/open-mmlab/mmsegmentation))

2.Put my model related files into the corresponding location of mmsegmentation according to the path I provided.