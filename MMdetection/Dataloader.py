# 모듈 import
from mmcv import Config
from mmdet.utils import get_device
import os

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def loadConfig(config, is_train):
    cfg = Config.fromfile(config['model_dir'])
    # dataset config 수정
    if is_train:
        cfg.device = get_device()
        cfg.data.train.classes = classes
        cfg.data.train.img_prefix = config['dataset_root']
        cfg.data.train.ann_file = os.path.join(config['dataset_root'], config['train_annotation']) # train json 정보
        cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize
        cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    else:
        cfg.data.test.test_mode = True
        cfg.model.train_cfg = None
        
    cfg.data.test.classes = tuple(config['classes'])
    cfg.data.test.img_prefix = config['dataset_root']
    cfg.data.test.ann_file = os.path.join(config['dataset_root'], config['test_annotation']) # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = config['seed']
    cfg.gpu_ids = [0]
    
    cfg.work_dir = os.path.join(config["work_dir"], config["exp_name"])
    
    cfg.model.roi_head.bbox_head.num_classes = len(config['classes'])

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    return cfg