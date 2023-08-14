_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_lars.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

crop_size = (768, 768)
ignore_idx = 255


# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 768), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=ignore_idx),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 768),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'LaRSDataset'
data_root = 'data/LaRS/v0.8.2/'

# LaRS unaugmented training set
dataset_lars_orig = dict(
    type='RepeatDataset',
    times=9, # To match the number of augmented samples
    dataset=dict(
        type=dataset_type,
        data_root=data_root + 'train',
        split='image_list.txt',
        pipeline=train_pipeline),
)

# LaRS augmented training set
dataset_lars_aug = dict(
    type=dataset_type,
    data_root=data_root + 'train_aug_obst_v2',
    pipeline=train_pipeline),

data = dict(
    samples_per_gpu=1, # NOTE: config made for 8 GPUs
    workers_per_gpu=1, # NOTE: config made for 8 GPUs
    train=[
        dataset_lars_orig,
        dataset_lars_aug
    ],
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)))
