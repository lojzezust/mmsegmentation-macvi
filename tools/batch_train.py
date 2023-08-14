import subprocess

CONFIGS=[
    # ("bn1_lars", "configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_lars.py"),
    # ("bn2_lars", "configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_lars.py"),
    # ("dl3_lars", "configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_lars.py"),
    # ("dl3p_lars", "configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_lars.py"),
    # ("unet_lars", "configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_lars.py"),
    # ("pointrend_lars", "configs/point_rend/pointrend_r101_512x1024_80k_lars.py"),
    # ("segformer_lars", "configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_lars.py"),
    # ("segmenter_lars", "configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_lars.py"),
    # ("stdc1_lars", "configs/stdc/stdc1_in1k-pre_512x1024_80k_lars.py"),
    # ("stdc2_lars", "configs/stdc/stdc2_in1k-pre_512x1024_80k_lars.py"),
    # ("fcn101_lars", "configs/fcn/fcn_r50-d8_512x1024_40k_lars.py"),
    # ("fcn50_lars", "configs/fcn/fcn_r101-d8_512x1024_40k_lars.py"),
    # ("knet_lars", "configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_lars.py"),
    # ("gena_genv5", "configs/segformer/segformer_mit-b2_8x1_768x768_160k_lars_genaug_v5.py"),
    # ("iccv_mastr", "configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_mastr1325.py")
    # ("iccv_mastr2", "configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_mastr1478.py")
    # ("iccv_rosebud", "configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_rosebud.py")
    ("iccv_mstr_rsbd", "configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_mastr_rosebud.py")
]

for task_name, cfg in CONFIGS:
    subprocess.run(['tools/slurm_train_my.sh', task_name, cfg])
