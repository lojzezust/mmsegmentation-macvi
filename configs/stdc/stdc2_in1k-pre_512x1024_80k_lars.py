checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
_base_ = './stdc1_in1k-pre_512x1024_80k_lars.py'
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            stdc_type='STDCNet2',
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
