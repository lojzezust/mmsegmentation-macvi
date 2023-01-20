_base_ = 'knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_lars.py'

data_root = 'data/LaRS/'
data = dict(
    train=dict(data_root=data_root + 'all'),
    test=dict(data_root=data_root + 'all')
)
