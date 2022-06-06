import copy as cp
import io
import os
import os.path as osp
import shutil
import warnings

import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..builder import PIPELINES
from .loading import SampleFrames


from statistics import NormalDist



@PIPELINES.register_module()
class GaussianSampleFrames(SampleFrames):
    def __init__(self, mean, sigma, clip_len, frame_interval=1, num_clips=1, temporal_jitter=False, twice_sample=False, out_of_bound_opt='loop', test_mode=False, start_index=None, keep_tail_frames=False):
        super().__init__(clip_len, frame_interval, num_clips, temporal_jitter, twice_sample, out_of_bound_opt, test_mode, start_index, keep_tail_frames)
        self.mean =mean
        self.sigma = sigma
        self.normalDist_sec = NormalDist(mean, sigma)
    
    def _get_train_clips(self, num_frames):
        _coeff = (self.normalDist.cdf(num_frames) - self.normalDist.cdf(0)) / self.num_clips
        _b = self.normalDist.cdf(0)
        base_offsets = [int(self.normalDist.inv_cdf(i * _coeff + _b)) for i in range(self.num_clips)]
        base_offsets += [num_frames]
        clip_offsets = np.random.randint(base_offsets[:-1], base_offsets[1:])
        return clip_offsets
    
    def _get_test_clips(self, num_frames):
        _coeff = (self.normalDist.cdf(num_frames) - self.normalDist.cdf(0)) / self.num_clips
        _b = self.normalDist.cdf(0)
        base_offsets = [int(self.normalDist.inv_cdf(i * _coeff + _b)) for i in range(self.num_clips)]
        base_offsets += [num_frames]
        base_offsets = np.array(base_offsets)
        clip_offsets = (base_offsets[:-1]  + base_offsets[1:]) / 2 
        return clip_offsets.astype(np.int)
    
    def __call__(self, results):
        self.rate = results['rate']
        self.normalDist = self.normalDist_sec * self.rate
        # print(self.normalDist)
        return super().__call__(results)
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                f'mean={self.mean}, '
                f'sigma={self.sigma}, '
                f'clip_len={self.clip_len}, '
                f'frame_interval={self.frame_interval}, '
                f'num_clips={self.num_clips}, '
                f'temporal_jitter={self.temporal_jitter}, '
                f'twice_sample={self.twice_sample}, '
                f'out_of_bound_opt={self.out_of_bound_opt}, '
                f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SelectROI:
    def __init__(self, roi_cfg) -> None:
        self.roi_cfg = roi_cfg
    
    
    def __call__(self, results):
        imgs = results['imgs']
        roi = self.roi_cfg[imgs[0].shape]
        img_roi = [img_cur[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                    for img_cur in imgs]
        results['imgs'] = img_roi
        # results['original_shape'] = img_roi[0].shape[:2]
        results['img_shape'] = img_roi[0].shape[:2]
        
        return results
        