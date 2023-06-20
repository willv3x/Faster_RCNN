import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def fasterrcnn_mobilenet_v3_large_fpn(num_classes, trainable_backbone_layers):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        trainable_backbone_layers=trainable_backbone_layers,  # default=3, min=0, max=6
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
