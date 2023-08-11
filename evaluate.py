import albumentations
import torch
from albumentations.pytorch import ToTensorV2

from data.data_loader import data_loader
from data.pascal_voc_dataset import PascalVOCDataset
from job.evaluator import Evaluator
from model.fasterrcnn_mobilenet_v3_large_320_fpn import fasterrcnn_mobilenet_v3_large_320_fpn
from model.fasterrcnn_mobilenet_v3_large_fpn import fasterrcnn_mobilenet_v3_large_fpn
from model.fasterrcnn_resnet50_fpn import fasterrcnn_resnet50_fpn
from model.fasterrcnn_resnet50_fpn_v2 import fasterrcnn_resnet50_fpn_v2

if __name__ == '__main__':
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CLASSES = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    NUM_CLASSES = len(CLASSES)

    NUM_WORKERS = 10

    #MODEL = fasterrcnn_resnet50_fpn_v2(NUM_CLASSES, 5)
    MODEL = fasterrcnn_resnet50_fpn(NUM_CLASSES, 5)
    # MODEL = fasterrcnn_mobilenet_v3_large_fpn(NUM_CLASSES, 6)
    # MODEL = fasterrcnn_mobilenet_v3_large_320_fpn(NUM_CLASSES, 6)

    MODEL.load_state_dict(torch.load('fasterrcnn_resnet50_fpn-ox_1080/train-ox_1080/best_map.pt', map_location=DEVICE))
    MODEL.to(DEVICE)

    TEST_AND_VALIDATION_TRANSFORM = albumentations.Compose([
        ToTensorV2()
    ], bbox_params=albumentations.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

    TEST_DATASET = PascalVOCDataset(
        directory_path='C:\ml\datasets\ox_real_1280_test.voc\\test',
        classes=CLASSES,
        transforms=TEST_AND_VALIDATION_TRANSFORM
    )

    TEST_LOADER = data_loader(TEST_DATASET, 1, True, NUM_WORKERS, True)

    EVALUATION_MAP = Evaluator().evaluate(MODEL, DEVICE, TEST_LOADER)

    Evaluator().plot_evaluation_map(EVALUATION_MAP, CLASSES)
