_base_ = [
    '../_base_/models/bisenetv1_r18-d32.py',
    '../_base_/datasets/lars.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(
            type='ResNet', depth=101,
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet101_v1c'))),
    decode_head=dict(in_channels=1024, channels=1024, num_classes=3, ignore_index=255),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=3,
            ignore_index=255,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=3,
            ignore_index=255,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ])
lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.005)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
