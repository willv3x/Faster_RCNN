import os

import albumentations
import torch
from albumentations.pytorch import ToTensorV2

from job.predictor import visualize_inferences
from model.fasterrcnn_resnet50_fpn_v2 import fasterrcnn_resnet50_fpn_v2

if __name__ == '__main__':

    CLASSES = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    NUM_CLASSES = len(CLASSES)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    MODEL = fasterrcnn_resnet50_fpn_v2(NUM_CLASSES, 5)
    MODEL.load_state_dict(
        torch.load('fasterrcnn_resnet50_fpn_v2-ox_1080/train-ox_1080/best_map.pt', map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    INFERENCE_TRANSFORM = albumentations.Compose([
        ToTensorV2()
    ])

    directory = 'C:\ml\datasets\ox_real_1280_test.yolov5pytorch\\test\images'

    images = []

    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            images.append(file_path)

    print(images)

    visualize_inferences(MODEL, DEVICE, images, INFERENCE_TRANSFORM, CLASSES)
