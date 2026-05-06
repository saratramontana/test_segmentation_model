import torch
import torch.nn as nn
import torchmetrics


def get_binary_cls_weights():
    """
    Class weights calcolati sul training set:
        benign: 39
        malignant: 70

    formula:
        total / (2 * count_i)
    """
    return torch.tensor([1.397, 0.779], dtype=torch.float32)


def build_common_losses(cls_weights):
    seg_criterion = nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)
    return seg_criterion, cls_criterion


def build_binary_classification_metrics():
    return nn.ModuleDict({
        "f1": torchmetrics.classification.BinaryF1Score(),
        "precision": torchmetrics.classification.BinaryPrecision(),
        "recall": torchmetrics.classification.BinaryRecall(),
        "auc": torchmetrics.AUROC(task="binary"),
    })


def build_multiclass_segmentation_metrics(num_classes=3):
    return nn.ModuleDict({
        "dice": torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        ),
        "iou": torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=num_classes,
            average="macro",
        ),
    })


def build_binary_segmentation_metrics():
    return nn.ModuleDict({
        "dice": torchmetrics.classification.BinaryF1Score(),
        "iou": torchmetrics.classification.BinaryJaccardIndex(),
    })


def build_ovamta_stage1_classification_metrics():
    return nn.ModuleDict({
        "f1": torchmetrics.classification.MulticlassF1Score(
            num_classes=3,
            average="macro",
        ),
        "precision": torchmetrics.classification.MulticlassPrecision(
            num_classes=3,
            average="macro",
        ),
        "recall": torchmetrics.classification.MulticlassRecall(
            num_classes=3,
            average="macro",
        ),
        "auc": torchmetrics.AUROC(
            task="multiclass",
            num_classes=3,
        ),
    })

