import copy
import torch
from detectron2.data import detection_utils as utils
# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
import detectron2.data.transforms as T
import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from Detectron2.Dataloader import *
from detectron2.utils.logger import setup_logger
setup_logger()

def main(config):
    loadData(config, True)
    cfg = loadConfig(config, True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

    def MyMapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], format='BGR')
            
        transform_list = []
        for aug in config['transform']['augmentations']:
            aug_class = getattr(T, aug["type"])  # A.HorizontalFlip 등으로 변환
            transform_list.append(aug_class(**aug["params"]))  # 매개변수 적용
            
        image, transforms = T.apply_transform_gens(transform_list, image)
            
        dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
            
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop('annotations')
                if obj.get('iscrowd', 0) == 0
            ]
            
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)
            
        return dataset_dict

    class MyTrainer(DefaultTrainer):
        
        @classmethod
        def build_train_loader(cls, cfg, sampler=None):
            return build_detection_train_loader(
            cfg, mapper = MyMapper, sampler = sampler
            )
        
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                os.makedirs(os.path.join(config['output_dir'], config['exp_name']), exist_ok = True)
                output_folder = os.path.join(config['output_dir'], config['exp_name'])
                
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
