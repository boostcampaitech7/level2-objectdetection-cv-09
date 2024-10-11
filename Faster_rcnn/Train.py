import os
import torch
# faster rcnn model이 포함된 library
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
from Faster_rcnn.Loss import Averager, collate_fn
from Faster_rcnn.Dataset import CustomDataset
from Faster_rcnn.Transform import get_train_transform

def train_fn(config, num_epochs, train_data_loader, optimizer, model, device):
    best_loss = 1000
    loss_hist = Averager()
    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = os.path.join(config['output_dir'], config["exp_name"], 'checkpoints.pth')
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value

def main(config):
    device = config['device']
    # 데이터셋 불러오기
    annotation = config['train_annotation'] # annotation 경로
    data_dir = config['data_dir'] # data_dir 경로
    
    train_dataset = CustomDataset(annotation, data_dir, True, get_train_transform(config['transform']["augmentations"])) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    device = torch.device(device)
    
    # torchvision model 불러오기
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=config['pretrained'])
    model = getattr(torchvision.models.detection, config['model_name'])(pretrained=config['pretrained'])
    num_classes = config['num_classes'] + 1 # class 개수= 10 + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = getattr(torch.optim, config['optimizer']['type'])(params, **config['optimizer']['params'])
    num_epochs = config['epoch']
    
    # training
    train_fn(config, num_epochs, train_data_loader, optimizer, model, device)