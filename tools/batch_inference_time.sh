#!/bin/sh

CONFIGS=(
    configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_lars.py
    configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_lars.py
    configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_lars.py
    configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_lars.py
    configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_lars.py
    configs/point_rend/pointrend_r101_512x1024_80k_lars.py
    configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_lars.py
    configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_lars.py
    configs/stdc/stdc1_in1k-pre_512x1024_80k_lars.py
    configs/stdc/stdc2_in1k-pre_512x1024_80k_lars.py
    configs/fcn/fcn_r50-d8_512x1024_40k_lars.py
    configs/fcn/fcn_r101-d8_512x1024_40k_lars.py
    configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_lars.py
)

for CONFIG in "${CONFIGS[@]}"
do
    echo $CONFIG
    python tools/inference_time.py $CONFIG
done
