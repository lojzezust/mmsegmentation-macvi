_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/lars_fixed.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(ignore_index=4, num_classes=3),
    auxiliary_head=dict(ignore_index=4, num_classes=3),
    # Model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
