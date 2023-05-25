import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class MaSTrDataset(CustomDataset):
    """MaSTr-type dataset.

    MaSTr dataset contains maritime scenes with obstacle, water and sky masks.
    """

    CLASSES = ('obstacle', 'water', 'sky')

    PALETTE = [[247, 195, 37], [41, 167, 224], [90, 75, 164]]

    def __init__(self, data_root=None, split=None, **kwargs):
        super(MaSTrDataset, self).__init__(
            data_root=data_root,
            split=split,
            img_dir='images',
            img_suffix='.jpg',
            ann_dir='masks',
            seg_map_suffix='m.png',
            ignore_index=4,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(data_root)
