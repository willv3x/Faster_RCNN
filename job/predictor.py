import cv2
import numpy as np
import torch
import albumentations
from albumentations.pytorch import ToTensorV2

INFERENCE_TRANSFORM = albumentations.Compose([
    ToTensorV2()
])




def visualize_inferences(model, device, images_paths, transform, detection_threshold=0.75):
    images = []
    for image_path in images_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255
        image = transform(image=image)['image']
        image = image.to(device)
        images.append(image)

    with torch.no_grad():
        outputs = model(images)

    predictions = []
    for output in outputs:
        boxes = output['boxes'].detach().cpu()
        labels = output['labels'].detach().cpu()
        scores = output['scores'].detach().cpu()

        boxes = boxes[scores >= detection_threshold]

        prediction = {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }
        predictions.append(prediction)

    for i, (image, prediction) in enumerate(zip(images, predictions)):
        image = drawBoxes(image, prediction)

        cv2.imwrite(f"predicted{i + 1}.jpg", image * 255.0)
