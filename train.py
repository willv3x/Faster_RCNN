import albumentations
import torch
from albumentations.pytorch import ToTensorV2

from data.data_loader import build_data_loader
from data.pascal_voc_dataset import PascalVOCDataset


def train():
    print(torch.cuda.is_available())

    TRAIN_TRANSFORM = albumentations.Compose([
        ToTensorV2()
    ], bbox_params=albumentations.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

    TEST_AND_VALIDATION_TRANSFORM = albumentations.Compose([
        ToTensorV2()
    ], bbox_params=albumentations.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

    INFERENCE_TRANSFORM = albumentations.Compose([
        ToTensorV2()
    ])

    TRAIN_DATASET = PascalVOCDataset(
        directory_path='C:\ml\datasets\glicosimetros-completo.v1i.voc\\train',
        classes=['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        transforms=TRAIN_TRANSFORM
    )

    print(f'Tamanho dataset de treino: {len(TRAIN_DATASET)}')

    data_loader = build_data_loader(TRAIN_DATASET, 2, True, 2, True)

    images, targets = next(iter(data_loader))

    print(targets)


if __name__ == '__main__':
    train()
