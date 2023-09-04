# LaRS Segmentation Starter Kit (MMSegmentation)

This repository is a fork of [MMSegmentation 0.x](https://github.com/open-mmlab/mmsegmentation/tree/0.x).

It provides a starting point for running semantic segmentation experiments on the LaRS dataset.
- Dataloader for LaRS
- Configs for a large number of segmentation methods
- Utilities for training and making predictions on LaRS

This document provides the basic information and steps to run simple training and inference tasks. For more complex use case scenarios, please refer to the [official MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation/tree/0.x).

## Installation

Follow the instructions to install this version of MMSegmentation.

**Step 1.** Clone the repository:
```shell
git clone https://github.com/lojzezust/mmsegmentation-macvi.git
cd mmsegmentation-macvi
```
**Step 1.** Create a conda or virtualenv environment. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

```shell
pip3 install torch torchvision
```

**Step 2.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```

**Step 3.** Install MMSegmentation (MaCVi) from source.

```shell
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 4.** Configure paths.

Download the [LaRS dataset](https://lojzezust.github.io/lars-dataset/). Update the path in the dataset config `configs/_base_/datasets/lars.py`, to point to the location of LaRS dataset.

## Getting started

### Training methods

Use one of the [provided training configs](#configs) to train a method.

```shell
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py configs/fcn/fcn_r50-d8_512x1024_40k_lars.py
```

By default the configs use a batch size of 4 per GPU. You can change this in the dataset config (`configs/_base_/datasets/lars.py`).

### Running inference

Use the `tools/test.py` script to run inference on the LaRS test set (for submission).

```shell
CONFIG=configs/fcn/fcn_r50-d8_512x1024_40k_lars.py
WEIGHTS=work_dirs/fcn_r50-d8_512x1024_40k_lars/latest.pth # Weights path
OUT_DIR=output/fcn_r50-d8_512x1024_40k_lars # Output dir

export CUDA_VISIBLE_DEVICES=0
python tools/test.py $CONFIG $WEIGHTS --show-dir $OUT_DIR
```

Use the `--val` flag to run on the validation set instead (for local evaluation).

```shell
python tools/test.py $CONFIG $WEIGHTS --show-dir $OUT_DIR --val
```

## Configs

The following LaRS configs are included in this repository:

| method     | backbone   | config                                                                  |
|------------|------------|-------------------------------------------------------------------------|
| FCN        | ResNet-50  | configs/fcn/fcn_r50-d8_512x1024_40k_lars.py                             |
| FCN        | ResNet-101 | configs/fcn/fcn_r101-d8_512x1024_40k_lars.py                            |
| UNet       | S5         | configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_lars.py                  |
| DeepLabv3  | ResNet-101 | configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_lars.py                |
| DeepLabv3+ | ResNet-101 | configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_lars.py        |
| BiSeNetv1  | ResNet-50  | configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_lars.py |
| BiSeNetv2  | -          | configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_lars.py              |
| STDC 1     | -          | configs/stdc/stdc1_in1k-pre_512x1024_80k_lars.py                        |
| STDC 2     | -          | configs/stdc/stdc2_in1k-pre_512x1024_80k_lars.py                        |
| PointRend  | ResNet-101 | configs/point_rend/pointrend_r101_512x1024_80k_lars.py                  |
| SegFormer  | MiT-B2     | configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_lars.py           |
| Segmenter  | ViT-B      | configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_lars.py         |
| KNet       | Swin-T     | configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_lars.py       |
