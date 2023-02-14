import subprocess
import os.path as osp

WORK_DIRS = 'work_dirs'
PRED_DIRS = 'output/predictions_v0.8'

CONFIGS=[
    ("bn1_lars_test", "configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_lars.py"),
    ("bn2_lars_test", "configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_lars.py"),
    ("dl3_lars_test", "configs/deeplabv3/deeplabv3_r101-d8_512x1024_40k_lars.py"),
    ("dl3p_lars_test", "configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_lars.py"),
    ("unet_lars_test", "configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_lars.py"),
    ("pointrend_lars_test", "configs/point_rend/pointrend_r101_512x1024_80k_lars.py"),
    ("segformer_lars_test", "configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_lars.py"),
    ("segmenter_lars_test", "configs/segmenter/segmenter_vit-b_mask_8x1_512x512_160k_lars.py"),
    ("stdc1_lars_test", "configs/stdc/stdc1_in1k-pre_512x1024_80k_lars.py"),
    ("stdc2_lars_test", "configs/stdc/stdc2_in1k-pre_512x1024_80k_lars.py"),
    ("fcn101_lars_test", "configs/fcn/fcn_r50-d8_512x1024_40k_lars.py"),
    ("fcn50_lars_test", "configs/fcn/fcn_r101-d8_512x1024_40k_lars.py"),
    ("knet_lars_test", "configs/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_lars.py")
]

for task_name, cfg in CONFIGS:
    run_name = osp.splitext(osp.basename(cfg))[0]
    checkpoint_pth = osp.join(WORK_DIRS, run_name, 'latest.pth')
    work_dir = osp.join(WORK_DIRS, run_name)
    pred_dir = osp.join(PRED_DIRS, run_name)
    subprocess.run(['tools/slurm_test_my.sh', task_name, cfg, checkpoint_pth, '--work-dir', work_dir,
                    '--eval',  'mIoU', '--show-dir', pred_dir, '--opacity', '1'])

