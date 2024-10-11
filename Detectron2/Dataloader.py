from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os

def loadData(config, is_train):
    if is_train:
        # Register Dataset
        MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
        try:
            register_coco_instances('coco_trash_train', {}, os.path.join(config['dataset_root'], config['train_annotation']), config['dataset_root'])
        except AssertionError:
            pass
         
    try:
        register_coco_instances('coco_trash_test', {}, os.path.join(config['dataset_root'], config['test_annotation']), config['dataset_root'])
    except AssertionError:
        pass


def loadConfig(config, is_train):
    # config 수정하기
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['model_url']))
    
    cfg.DATASETS.TEST = ('coco_trash_test',)
    cfg.DATALOADER.NUM_WOREKRS = 2
    cfg.OUTPUT_DIR = os.path.join(config['output_dir'], config['exp_name'])
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['parameters']['batch_size_per_image']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    
    if is_train:
        cfg.DATASETS.TRAIN = ('coco_trash_train',)
        cfg.SOLVER.IMS_PER_BATCH = config['parameters']['ims_per_batch']
        cfg.SOLVER.BASE_LR = config['parameters']['base_lr']
        cfg.SOLVER.MAX_ITER = config['parameters']['max_iter']
        cfg.SOLVER.STEPS = tuple(config['parameters']['steps'])
        cfg.SOLVER.GAMMA = config['parameters']['gamma']
        cfg.SOLVER.CHECKPOINT_PERIOD = config['parameters']['checkpoint_period']
        cfg.TEST.EVAL_PERIOD = config['parameters']['eval_period']
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['model_url'])
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['parameters']['score_thresh_test']
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    return cfg