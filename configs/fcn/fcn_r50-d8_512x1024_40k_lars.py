_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/lars.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

model = dict(
    decode_head=dict(ignore_index=255, num_classes=3),
    auxiliary_head=dict(ignore_index=255, num_classes=3),
)
