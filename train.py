import albumentations
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot
from torch.utils.data import DataLoader

from data.pascal_voc_dataset import PascalVOCDataset

print(torch.cuda.is_available())

# TRAIN_DATASET = PascalVOCDataset(
#     images_folder=os.path.join(os.path.abspath(''), 'glicosimetros.v1i.coco/train'),
#     annotation_file=os.path.join(os.path.abspath(''), 'glicosimetros.v1i.coco/train/_annotations.coco.json')
# )

# VALIDATION_DATASET = CocoDataset(
#     images_folder = os.path.join(os.path.abspath(''), 'glicosimetros.v1i.coco/valid'),
#     annotation_file = os.path.join(os.path.abspath(''), 'glicosimetros.v1i.coco/valid/_annotations.coco.json'),
#     transforms = TEST_AND_VALIDATION_TRANSFORM
# )

# TEST_DATASET = CocoDataset(
#     images_folder = os.path.join(os.path.abspath(''), 'glicosimetros.v1i.coco/test'),
#     annotation_file = os.path.join(os.path.abspath(''), 'glicosimetros.v1i.coco/test/_annotations.coco.json'),
#     transforms = TEST_AND_VALIDATION_TRANSFORM
# )

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

VALIDATION_DATASET = PascalVOCDataset(
    images_folder='C:\ml\datasets\glicosimetros.v1i.coco\\valid',
    annotation_file='C:\ml\datasets\glicosimetros.v1i.coco\\valid\_annotations.coco.json',
    transforms=TEST_AND_VALIDATION_TRANSFORM
)

TEST_DATASET = PascalVOCDataset(
    images_folder='C:\ml\datasets\glicosimetros.v1i.coco\\test',
    annotation_file='C:\ml\datasets\glicosimetros.v1i.coco\\test\_annotations.coco.json',
    transforms=TEST_AND_VALIDATION_TRANSFORM
)

CLASSES = list(category['name'] for category in TRAIN_DATASET.get_categories_dict().values())
NUM_CLASSES = len(CLASSES)


print(f'Tamanho dataset de validação: {len(VALIDATION_DATASET)}')
print(f'Tamanho dataset de teste: {len(TEST_DATASET)}')


def get_train_loader(batch_size, num_workers):
    return DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                      collate_fn=lambda batch: tuple(zip(*batch)))


def get_validation_loader(batch_size, num_workers):
    return DataLoader(VALIDATION_DATASET, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                      collate_fn=lambda batch: tuple(zip(*batch)))


def get_test_loader(batch_size, num_workers):
    return DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                      collate_fn=lambda batch: tuple(zip(*batch)))


def draw_boxes(image, target):
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
        label_name = CLASSES[label]

        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        ((label_width, label_height), _) = cv2.getTextSize(label_name, label_font, label_thickness, 1)

        cv2.rectangle(image_numpy, (x_min, y_min), (x_max, y_max), box_color, thickness=box_thickness)
        cv2.rectangle(image_numpy, (x_min, y_min - int(1.3 * label_height)), (x_min + label_width, y_min), box_color,
                      -1)
        cv2.putText(image_numpy, text=label_name, org=(x_min, y_min - int(0.3 * label_height)),
                    fontFace=label_font, fontScale=label_font_scale, color=label_color, thickness=1,
                    lineType=cv2.LINE_AA)

    return image_numpy


def visualize_some(dataset, num_images=4, num_col=2):
    fig = pyplot.figure(figsize=(25, 30))
    pyplot.axis('off')

    num_row = int(num_images / num_col) + num_images % num_col

    data_loader = DataLoader(dataset, batch_size=num_images, shuffle=True, num_workers=0, drop_last=True,
                             collate_fn=lambda batch: tuple(zip(*batch)))
    images, targets = next(iter(data_loader))

    for i, (image, target) in enumerate(zip(images, targets)):
        # image = REVERT_NORMALIZATION(image)
        image = draw_boxes(image, target)

        ax = fig.add_subplot(num_row, num_col, i + 1, xticks=[], yticks=[])
        pyplot.imshow(image)


visualize_some(TRAIN_DATASET)
