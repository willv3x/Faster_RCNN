import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_r_cnn_resnet50_fpn_v2(num_classes, trainable_backbone_layers):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        trainable_backbone_layers=trainable_backbone_layers,  # default=3, minimo=0, maximo=5
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
