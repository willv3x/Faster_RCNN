import cv2
import numpy as np
import torch


def draw_boxes(image, target, classes):
    box_thickness = 1
    box_color = (255, 0, 0)
    label_thickness = 0.7
    label_font_scale = 0.7
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (255, 255, 255)

    image_numpy = image.permute(1, 2, 0).cpu().numpy()

    labels = target['labels']
    boxes = target['boxes']

    for label, box in zip(labels, boxes):
        label = int(label)
        label_name = classes[label]

        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        ((label_width, label_height), _) = cv2.getTextSize(label_name, label_font, label_thickness, 1)

        cv2.rectangle(image_numpy, (x_min, y_min), (x_max, y_max), box_color, thickness=box_thickness)
        cv2.rectangle(image_numpy, (x_min, y_min - int(1.3 * label_height)), (x_min + label_width, y_min), box_color, -1)
        cv2.putText(image_numpy, text=label_name, org=(x_min, y_min - int(0.3 * label_height)),
                    fontFace=label_font, fontScale=label_font_scale, color=label_color, thickness=1, lineType=cv2.LINE_AA)

    return image_numpy


def visualize_inferences(model, device, images_paths, transform, classes, detection_threshold=0.75):
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
        image = draw_boxes(image, prediction, classes)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"predicted{i + 1}.jpg", image * 255.0)
