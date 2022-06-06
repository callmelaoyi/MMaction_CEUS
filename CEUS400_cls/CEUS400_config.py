_base_ = ['../configs/_base_/models/tsn_r50.py', '../configs/_base_/default_runtime.py']

model = dict(cls_head=dict(num_classes=3, init_std=0.001,topk=1))
model = dict(backbone=dict(partial_bn=True,))
dataset_type = 'CEUSDatsaset'
data_root = '.'
data_root_val = '.'
ann_file_train = f'data/CEUS400/CEUS400_train.txt'
ann_file_val = f'data/CEUS400/CEUS400_test.txt'
ann_file_test = f'data/CEUS400/CEUS400_test.txt'
gaussian_cfg = dict(mean=20,sigma=5)
roi_cfg = {(600, 800, 3): (69, 81, 338, 485), 
           (1080, 1440, 3): (15, 164, 683, 779), 
           (910, 1260, 3): (552, 130, 564, 698), 
           (768, 1024, 3): (37, 78, 484, 602)}
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='GaussianSampleFrames', **gaussian_cfg, clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='SelectROI', roi_cfg=roi_cfg),
    dict(type='Resize', scale=(-1, 256)),

    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='GaussianSampleFrames', **gaussian_cfg, clip_len=1, frame_interval=1, num_clips=16,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='SelectROI', roi_cfg=roi_cfg),
    dict(type='Resize', scale=(-1, 256)),
    
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='GaussianSampleFrames', **gaussian_cfg, clip_len=1, frame_interval=1, num_clips=16,
         test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='SelectROI', roi_cfg=roi_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=5, metrics=['mean_class_accuracy', 'confusion_matrix'])

optimizer = dict(
    type='SGD', lr=5e-3, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 75

# runtime settings
checkpoint_config = dict(interval=5)
load_from = "CEUS400_cls/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth"
# work_dir = f'./work_dirs/tsn_r50_1x1x3_75e_ucf101_split_{split}_rgb/'
