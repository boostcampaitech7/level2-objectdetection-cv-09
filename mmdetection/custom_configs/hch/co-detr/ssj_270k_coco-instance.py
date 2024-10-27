_base_ = ['./default_runtime.py', './scheduler.py']
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../dataset'

image_size = (768, 768)
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None







# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.25),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# LSJ + CopyPaste
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
]

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler'),
#     dataset=dict(
#         type=dataset_type,
#         metainfo=dict(classes=classes),
#         data_root=data_root,
#         ann_file=data_root + '/train.json',
#         data_prefix=dict(img=''),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline,
#         backend_args=backend_args))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file=data_root + '/train.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=load_pipeline,
            backend_args=backend_args),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=data_root + '/valid.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/valid.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)


test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file=data_root + '/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + '/test.json',
    outfile_prefix='./results/co_detr/test')

# The model is trained by 270k iterations with batch_size 64,
# which is roughly equivalent to 144 epochs.


