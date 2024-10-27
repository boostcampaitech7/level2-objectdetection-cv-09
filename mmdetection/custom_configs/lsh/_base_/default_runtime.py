default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

evaluation = dict(
    interval=1,  # 몇 번의 epoch마다 평가할 것인지
    metric='bbox',  # 사용할 평가 지표
    save_best='bbox_mAP_50',  # 'bbox_mAP_50' 등 최적의 성능에 따라 checkpoint 저장
    rule='greater'  # mAP가 클수록 좋으므로 'greater'를 사용 ('loss'라면 'less'를 사용)
)

checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=False)

vis_backends = [dict(type='WandbVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
