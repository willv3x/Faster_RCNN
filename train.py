import albumentations
import torch
import wandb
from albumentations.pytorch import ToTensorV2

from data.data_loader import data_loader
from data.pascal_voc_dataset import PascalVOCDataset
from job.trainer import Trainer
from model.fasterrcnn_mobilenet_v3_large_320_fpn import fasterrcnn_mobilenet_v3_large_320_fpn
from model.fasterrcnn_mobilenet_v3_large_fpn import fasterrcnn_mobilenet_v3_large_fpn
from model.fasterrcnn_resnet50_fpn import fasterrcnn_resnet50_fpn
from model.fasterrcnn_resnet50_fpn_v2 import fasterrcnn_resnet50_fpn_v2

if __name__ == '__main__':
    torch.manual_seed(42)
    # WANDB_ENTITY = "ah-visao"

    # PROJECT = 'fasterrcnn_resnet50_fpn_v2'
    # PROJECT = 'fasterrcnn_resnet50_fpn'
    # PROJECT = 'fasterrcnn_mobilenet_v3_large_fpn'
    PROJECT = 'fasterrcnn_mobilenet_v3_large_320_fpn'

    NAME = 'train-gl_full'

    wandb.login()

    LEARN_RATE = 0.003
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    EPOCHS = 150
    BATCH_SIZE = 2
    NUM_WORKERS = 10
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CLASSES = ['__background__', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    NUM_CLASSES = len(CLASSES)

    # BACKBONE_TRAINABLE_LAYERS = 5
    # MODEL = fasterrcnn_resnet50_fpn_v2(NUM_CLASSES, BACKBONE_TRAINABLE_LAYERS)
    # MODEL = fasterrcnn_resnet50_fpn(NUM_CLASSES, BACKBONE_TRAINABLE_LAYERS)
    BACKBONE_TRAINABLE_LAYERS = 6
    # MODEL = fasterrcnn_mobilenet_v3_large_fpn(NUM_CLASSES, BACKBONE_TRAINABLE_LAYERS)
    MODEL = fasterrcnn_mobilenet_v3_large_320_fpn(NUM_CLASSES, BACKBONE_TRAINABLE_LAYERS)

    MODEL.to(DEVICE)
    MODEL_PARAMETERS = [p for p in MODEL.parameters() if p.requires_grad]
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

    TRAIN_DATASET = PascalVOCDataset(
        directory_path='C:\ml\datasets\gl_full.voc\\train',
        classes=CLASSES,
        transforms=TRAIN_TRANSFORM
    )

    VALIDATION_DATASET = PascalVOCDataset(
        directory_path='C:\ml\datasets\gl_full.voc\\valid',
        classes=CLASSES,
        transforms=TEST_AND_VALIDATION_TRANSFORM
    )

    TRAIN_LOADER = data_loader(TRAIN_DATASET, BATCH_SIZE, True, NUM_WORKERS, True)
    VALIDATION_LOADER = data_loader(VALIDATION_DATASET, BATCH_SIZE, True, NUM_WORKERS, True)

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
        "workers": NUM_WORKERS,
    }

    RUN = wandb.init(
        # entity=WANDB_ENTITY,
        project=PROJECT,
        name=NAME,
        config=WANDB_CONFIG,
    )

    wandb.watch(MODEL, log_freq=10)

    print(f"\n\nIniciando treinamento usando PyTorch {torch.__version__} e dispositivo "
          f"{torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'}\n")

    epoch_train_losses, epoch_validation_metrics, log = \
        Trainer().train(MODEL, EPOCHS, DEVICE, OPTIMIZER, TRAIN_LOADER, VALIDATION_LOADER)

    RUN_ARTIFACT = wandb.Artifact(wandb.run.name, type='model')
    RUN_ARTIFACT.add_file('best_loss.pt')
    RUN_ARTIFACT.add_file('best_map.pt')
    RUN.log_artifact(RUN_ARTIFACT)
