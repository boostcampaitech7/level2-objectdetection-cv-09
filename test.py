import torch, gc
import numpy as np
import argparse
from config_parser import ConfigParser
import random
import Faster_rcnn.Inference
import MMdetection.Inference
import Detectron2.Inference

def main(config, config_path):
    gc.collect()
    torch.cuda.empty_cache()
    # Load datasets
    if config['model_type'] == 'torchvision':
        Faster_rcnn.Inference.main(config)
    if config['model_type'] == 'mmdetection':
        MMdetection.Inference.main(config)
    if config['model_type'] == 'detectron2':
        Detectron2.Inference.main(config)
    # 학습 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-c', '--config', type=str, required=True, 
                        help="Path to the configuration YAML file")

    args = parser.parse_args()

    config_parser = ConfigParser(args.config)
    config = config_parser.config
    
    # fix random seeds for reproducibility
    SEED = config['seed']
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    main(config = config, config_path = args.config)