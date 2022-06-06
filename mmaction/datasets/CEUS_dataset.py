import copy
import os.path as osp

import torch

from mmaction.datasets.pipelines import Resize
from .base import BaseDataset
from .builder import DATASETS
from . import RawframeDataset




@DATASETS.register_module()
class CEUSDatsaset(RawframeDataset):
    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False, filename_tmpl='img_{:05}.jpg', with_offset=False, multi_class=False, num_classes=None, start_index=1, modality='RGB', sample_by_class=False, power=0, dynamic_length=False, **kwargs):
        super().__init__(ann_file, pipeline, data_prefix, test_mode, filename_tmpl, with_offset, multi_class, num_classes, start_index, modality, sample_by_class, power, dynamic_length, **kwargs)
    
    
    
    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                video_info['frame_dir'] = line_split[0]
                video_info['total_frames'] = int(line_split[1])
                video_info['label'] = int(line_split[2])
                video_info['rate'] = float(line_split[3])
                video_infos.append(video_info)
        return video_infos