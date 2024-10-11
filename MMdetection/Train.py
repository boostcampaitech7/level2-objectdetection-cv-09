# 모듈 import
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from MMdetection.Dataloader import loadConfig

def main(config):
    cfg = loadConfig(config, True)
    
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.init_weights()
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)