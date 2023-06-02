import glob
import os
from xml.etree import ElementTree

import cv2
import numpy
import torch
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):
    def __init__(self, directory_path, classes, transforms=None):
        self._directory_path = directory_path
        self._classes = classes
        self._transforms = transforms

        self._images_paths = glob.glob(f"{self._directory_path}/*.jpg")
        self._images_file_names = [image_path.split(os.path.sep)[-1] for image_path in self._images_paths]
        self._images_file_names = sorted(self._images_file_names)

        print(len(self._images_file_names))

    def __getitem__(self, index):
        image_file_name = self._images_file_names[index]
        image_file_path = os.path.join(self._directory_path, image_file_name)

        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(numpy.float32) / 255

        annotation_file_name = image_file_name[:-4] + '.xml'
        annotation_file_path = os.path.join(self._directory_path, annotation_file_name)

        annotations = ElementTree.parse(annotation_file_path).getroot()
        annotations_data = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
        }

        for annotation in annotations.findall('object'):
            x_min = int(annotation.find('bndbox').find('xmin').text)
            x_max = int(annotation.find('bndbox').find('xmax').text)
            y_min = int(annotation.find('bndbox').find('ymin').text)
            y_max = int(annotation.find('bndbox').find('ymax').text)

            label = self._classes.index(annotation.find('name').text)
            box = [x_min, y_min, x_max, y_max]
            area = (x_max - x_min) * (y_max - y_min)
            iscrowd = 0

            annotations_data['labels'].append(label)
            annotations_data['boxes'].append(box)
            annotations_data['area'].append(area)
            annotations_data['iscrowd'].append(iscrowd)

        if self._transforms is not None:
            transformed = self._transforms(image=image, bboxes=annotations_data['boxes'],
                                           labels=annotations_data['labels'])
            image = transformed['image']
            annotations_data['boxes'] = transformed['bboxes']
            annotations_data['labels'] = transformed['labels']

        target = {
            'image_id': torch.tensor([index]),
            'boxes': torch.as_tensor(annotations_data['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(annotations_data['labels'], dtype=torch.int64),
            'area': torch.as_tensor(annotations_data['area'], dtype=torch.float32),
            'iscrowd': torch.as_tensor(annotations_data['iscrowd'], dtype=torch.int64)
        }
        return image, target

    def __len__(self):
        return len(self._images_file_names)
