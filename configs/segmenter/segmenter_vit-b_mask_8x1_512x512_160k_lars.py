_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/lars.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
optimizer = dict(lr=0.001, weight_decay=0.0)

# Override
model = dict(
    # backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=3, ignore_index=4)
)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    # num_gpus: 2 -> batch_size: 8
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline))
