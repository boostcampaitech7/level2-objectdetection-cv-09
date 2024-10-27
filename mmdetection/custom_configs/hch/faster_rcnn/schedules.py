# training schedule for 1x

max_epoch = 10

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=0.5,
#         begin=0,
#         T_max=max_epoch,
#         end=max_epoch,
#         by_epoch=True,
#         convert_to_iter_based=False)
# ]

# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0001,
#         weight_decay=0.0001,
#         eps=1e-8,
#         betas=(0.9, 0.999)))


# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
