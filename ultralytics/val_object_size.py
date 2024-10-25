from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def evaluate_custom_dataset(annotation_file, prediction_file):
    # 1. COCO 형식의 annotation (라벨) 파일 로드
    coco = COCO(annotation_file)

    # 2. COCO 형식의 prediction (예측 결과) 파일 로드
    coco_pred = coco.loadRes(prediction_file)

    # 3. COCOeval 객체 생성 (평가 지표: 'bbox' 사용)
    coco_eval = COCOeval(coco, coco_pred, 'bbox')

    # 4. 평가 실행
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 5. 크기별 mAP 출력
    print(f"mAP@0.5 (IoU=0.5): {coco_eval.stats[0]:.3f}")
    print(f"mAP@0.5:0.95 (IoU=0.5:0.95): {coco_eval.stats[1]:.3f}")
    print(f"Small objects mAP@0.5:0.95: {coco_eval.stats[3]:.3f}")  # Small objects mAP
    print(f"Medium objects mAP@0.5:0.95: {coco_eval.stats[4]:.3f}")  # Medium objects mAP
    print(f"Large objects mAP@0.5:0.95: {coco_eval.stats[5]:.3f}")  # Large objects mAP

# 예시 실행
annotation_file = '/home/wook/yehna/project2/dataset_split/valid.json'  # COCO 형식의 annotation 파일 경로
prediction_file = '/home/wook/yehna/project2/ultralytics/runs/detect/val/predictions.json'  # COCO 형식의 예측 파일 경로

evaluate_custom_dataset(annotation_file, prediction_file)