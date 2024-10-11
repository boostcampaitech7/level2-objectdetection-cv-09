import os
from tqdm import tqdm
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from Detectron2.Dataloader import *
from Detectron2.Dataset import *
    
def MyMapper(dataset_dict):
        
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
        
    dataset_dict['image'] = image
    
    return dataset_dict
    
def main(config):
    loadData(config, False)
    cfg = loadConfig(config, False)
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', MyMapper)
        # output 뽑은 후 sumbmission 양식에 맞게 후처리 
    prediction_strings = []
    file_names = []

    class_num = 10

    for data in tqdm(test_loader):
        
        prediction_string = ''
        
        data = data[0]
        
        outputs = predictor(data['image'])['instances']
        
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace(config['dataset_root'],''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, 'submission.csv'), index=None)