import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(transform_config):
    aug_list = []
    for aug in transform_config:
        aug_class = getattr(A, aug["type"])  # A.HorizontalFlip 등으로 변환
        aug_list.append(aug_class(**aug["params"]))  # 매개변수 적용
    aug_list += [ToTensorV2(p = 1.0)]
    return A.Compose(aug_list, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})