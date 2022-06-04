from mmaction.datasets import build_dataset
from mmcv import Config, DictAction



cfg = Config.fromfile('CICV_CEUS_config.py')

# dataset = build_dataset(cfg.data.train)
dataset = build_dataset(cfg.data.val)

print(len(dataset))
for i in dataset:
    print(i['imgs'].shape)