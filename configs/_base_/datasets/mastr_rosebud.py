# Merges MaSTr1325, MaSTr153, and ROSEBUD datasets
dataset_type = 'MaSTrDataset'
mastr_data_root = 'data/mastr1325/'
ad_data_root = 'data/mastr153/'
rosebud_data_root = 'data/ROSEBUD/'

ignore_idx=4
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
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
        img_scale=(2048, 1024),
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

mastr153_repeat = dict(
    type='RepeatDataset',
    times=8,
    dataset=dict(
        type=dataset_type,
        data_root=ad_data_root,
        split='annotated_list.txt',
        pipeline=train_pipeline)
)

mastr1325_train = dict(
    type=dataset_type,
    data_root=mastr_data_root,
    split='all_list.txt',
    pipeline=train_pipeline)

rosebud_train = dict(
    type=dataset_type,
    data_root=rosebud_data_root,
    img_suffix='.png',
    seg_map_suffix='.png',
    split=None,
    pipeline=train_pipeline),

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[mastr1325_train, mastr153_repeat, rosebud_train],
    val=dict(
        type=dataset_type,
        data_root=mastr_data_root,
        split='val_list.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=mastr_data_root,
        split='val_list.txt',
        pipeline=test_pipeline))
