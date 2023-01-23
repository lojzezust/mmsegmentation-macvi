import argparse
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from tqdm.auto import tqdm
from time import time
import numpy as np
import os
import os.path as osp
import json

import mmcv
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_dp, get_device

NUM_SAMPLES = 200
NUM_REPEATS = 5
OUTPUT_DIR = 'output/latency'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the model configuration file')
    parser.add_argument('-n', '--num_samples', type=int, default=NUM_SAMPLES,
                        help='Number of data samples used in the test.')
    parser.add_argument('-r', '--repeats', type=int, default=NUM_REPEATS,
                        help='Number of repetitions.')
    parser.add_argument('-o', '--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory.')

    args = parser.parse_args()
    return args

def eval_inference_time(args):
    cfg = mmcv.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Build dataset
    dataset = build_dataset(cfg.data.test)

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # Build model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Prepare for inference
    torch.cuda.empty_cache()
    model = revert_sync_batchnorm(model)
    model = build_dp(model, get_device(), device_ids=[0])
    model.eval()

    # Preload data
    data = []
    for i,d in tqdm(enumerate(data_loader), desc='Reading data', total=args.num_samples):
        if i >= args.num_samples:
            break
        data.append(d)

    with torch.no_grad():
        # Network warm-up
        for d in tqdm(data, desc='Network warm-up'):
            model(return_loss=False, **d)

        # Time inference
        times = []
        for it in range(args.repeats):
            start_time = time()
            for d in tqdm(data, desc='Running inference %d/%d' % (it+1, args.repeats)):
                model(return_loss=False, **d)

            end_time = time()
            elapsed = end_time - start_time
            times.append(elapsed)

    # Compute and print statistics
    times = np.array(times)
    per_img = times / args.num_samples
    fps = 1. / per_img

    per_img_m = per_img.mean()
    per_img_std = per_img.std()

    fps_m = fps.mean()
    fps_std = fps.std()

    print('Latency: %.0f ± %.1f ms, FPS: %.1f ± %.2f' % (per_img_m * 1000, 2 * per_img_std * 1000, fps_m, 2 * fps_std))

    summary = {
        'num_samples': args.num_samples,
        'times': times.tolist(),
        'latency': {'mean': per_img_m, 'std': per_img_std},
        'fps': {'mean': fps_m, 'std': fps_std},
    }

    # Save summary to JSON
    if not osp.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    cfg_name = osp.splitext(osp.basename(args.config))[0]
    with open(osp.join(OUTPUT_DIR, cfg_name + '.json'), 'w') as file:
        json.dump(summary, file)


if __name__ == '__main__':
    args = get_args()
    eval_inference_time(args)
