from mmaction.datasets import build_dataset
from mmcv import Config, DictAction



cfg = Config.fromfile('CEUS400_cls/CEUS400_config.py')

# dataset = build_dataset(cfg.data.train)
dataset = build_dataset(cfg.data.val)

print(len(dataset))
for i in dataset:
    print(i['imgs'].shape)