dataset_type = 'BaseSegDataset'
data_root = 'data/TTPLA'

crop_size = (512, 512)
train_scale = (2048, 1024)
test_scale = (1024, 512)

metainfo = dict(
    classes=('background', 'powerline'),
    palette=[[0, 0, 0], [255, 255, 255]],
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=train_scale, ratio_range=(0.8, 1.25), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.85, ignore_index=255),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_scale, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_list.txt',
        data_prefix=dict(
            img_path='imgs',
            seg_map_path='masks',
        ),
        metainfo=metainfo,
        pipeline=train_pipeline,
        img_suffix='',
        seg_map_suffix='',
        reduce_zero_label=False,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_list.txt',
        data_prefix=dict(
            img_path='imgs',
            seg_map_path='masks',
        ),
        metainfo=metainfo,
        pipeline=test_pipeline,
        img_suffix='',
        seg_map_suffix='',
        reduce_zero_label=False,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    class_names=['background', 'powerline'],
    nan_to_num=0,
    ignore_index=255,
    per_class_results=True,
)

test_evaluator = val_evaluator