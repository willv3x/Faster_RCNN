import albumentations
import torch
import wandb
from albumentations.pytorch import ToTensorV2

from data.data_loader import data_loader
from data.eager_pascal_voc_dataset import EagerPascalVOCDataset
from job.trainer import Trainer
from model.fasterrcnn_resnet50_fpn_v2 import fasterrcnn_resnet50_fpn_v2

if __name__ == '__main__':
    torch.manual_seed(42)
    WANDB_ENTITY = "ah-visao"
    MODEL_NAME = 'faster_r_cnn_resnet50_fpn_v2'

    wandb.login()

    CLASSES = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    NUM_CLASSES = len(CLASSES)
    EPOCHS = 60
    BATCH_SIZE = 2
    NUM_WORKERS = 5
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BACKBONE_TRAINABLE_LAYERS = 5
    MODEL = fasterrcnn_resnet50_fpn_v2(NUM_CLASSES, BACKBONE_TRAINABLE_LAYERS)
    MODEL.to(DEVICE)
    MODEL_PARAMETERS = [p for p in MODEL.parameters() if p.requires_grad]
    LEARN_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    OPTIMIZER = torch.optim.SGD(MODEL_PARAMETERS, lr=LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

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

    TRAIN_DATASET = EagerPascalVOCDataset(
        directory_path='C:\ml\datasets\oximetro.v1i.voc\\train',
        classes=CLASSES,
        transforms=TRAIN_TRANSFORM
    )

    VALIDATION_DATASET = EagerPascalVOCDataset(
        directory_path='C:\ml\datasets\oximetro.v1i.voc\\valid',
        classes=CLASSES,
        transforms=TEST_AND_VALIDATION_TRANSFORM
    )

    TRAIN_LOADER = data_loader(TRAIN_DATASET, BATCH_SIZE, True, NUM_WORKERS, True)
    VALIDATION_LOADER = data_loader(VALIDATION_DATASET, 1, True, NUM_WORKERS, True)

    WANDB_CONFIG = {
        "learn_rate": LEARN_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "backbone_trainable_layers": BACKBONE_TRAINABLE_LAYERS,
        "trainable_parameters_count": len(MODEL_PARAMETERS),
    }

    RUN = wandb.init(
        project=MODEL_NAME,
        entity=WANDB_ENTITY,
        config=WANDB_CONFIG,
    )

    wandb.watch(MODEL, log_freq=10)

    print(f"\n\nIniciando treinamento usando PyTorch {torch.__version__} e dispositivo "
          f"{torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'}\n")

    epoch_train_losses, epoch_validation_metrics, log = \
        Trainer().train(MODEL, EPOCHS, DEVICE, OPTIMIZER, TRAIN_LOADER, VALIDATION_LOADER)

    RUN_ARTIFACT = wandb.Artifact(wandb.run.name, type='model')
    RUN_ARTIFACT.add_file('best_train_model.pt')
    RUN_ARTIFACT.add_file('best_validation_model.pt')
    RUN.log_artifact(RUN_ARTIFACT)

