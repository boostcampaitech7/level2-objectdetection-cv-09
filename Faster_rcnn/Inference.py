from pycocotools.coco import COCO
import torch, os
# faster rcnn model이 포함된 library
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from Faster_rcnn.Dataset import CustomDataset

def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

def main(config):
    device = config['device']
    annotation = config['test_annotation'] # annotation 경로
    data_dir = config['data_dir'] # dataset 경로
    test_dataset = CustomDataset(annotation, data_dir, False)
    score_threshold = config['score_threshold']
    check_point = os.path.join(config['output_dir'], config["exp_name"], 'checkpoints.pth') # 체크포인트 경로
    

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,
        num_workers=4
    )
    device = torch.device(device)
    
    # torchvision model 불러오기
    model = getattr(torchvision.models.detection, config['model_name'])(pretrained=config['pretrained'])
    num_classes = config['num_classes'] + 1  # 10 class + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(check_point))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    submission.to_csv(os.path.join(config['output_dir'], config["exp_name"], 'submission.csv'), index=None)