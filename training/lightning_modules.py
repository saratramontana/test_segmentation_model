import os
import sys

PROJECT_ROOT = "/Users/saratramontana/Documents/GitHub/test_segmentation_model"

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "external", "ACSNet"))
sys.path.append(os.path.join(PROJECT_ROOT, "external", "OvaMTA"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import wandb
import torchvision
import segmentation_models_pytorch as smp
import torchmetrics

from MyModel import MyModel

from lib.OvaMTA_seg import TransRaUNet_CLF_xiaorong
from lib.OvaMTA_diag import TransRaUNet_CLF_xiaorong as OvaDiagModel
from utils.smooth_l1_loss import SmoothL1Loss

from training.metrics import (
    get_binary_cls_weights,
    build_common_losses,
    build_binary_classification_metrics,
    build_multiclass_segmentation_metrics,
    build_binary_segmentation_metrics,
    build_ovamta_stage1_classification_metrics,
)


def _assign_binary_cls_metrics(module):
    train_cls_metrics = build_binary_classification_metrics()
    val_cls_metrics = build_binary_classification_metrics()

    module.train_cls_f1 = train_cls_metrics["f1"]
    module.train_cls_precision = train_cls_metrics["precision"]
    module.train_cls_recall = train_cls_metrics["recall"]
    module.train_cls_auc = train_cls_metrics["auc"]

    module.val_cls_f1 = val_cls_metrics["f1"]
    module.val_cls_precision = val_cls_metrics["precision"]
    module.val_cls_recall = val_cls_metrics["recall"]
    module.val_cls_auc = val_cls_metrics["auc"]


def _assign_multiclass_seg_metrics(module):
    train_seg_metrics = build_multiclass_segmentation_metrics(num_classes=3)
    val_seg_metrics = build_multiclass_segmentation_metrics(num_classes=3)

    module.train_seg_dice = train_seg_metrics["dice"]
    module.train_seg_iou = train_seg_metrics["iou"]

    module.val_seg_dice = val_seg_metrics["dice"]
    module.val_seg_iou = val_seg_metrics["iou"]


def _assign_binary_seg_metrics(module):
    train_seg_metrics = build_binary_segmentation_metrics()
    val_seg_metrics = build_binary_segmentation_metrics()

    module.train_seg_dice = train_seg_metrics["dice"]
    module.train_seg_iou = train_seg_metrics["iou"]

    module.val_seg_dice = val_seg_metrics["dice"]
    module.val_seg_iou = val_seg_metrics["iou"]


def _assign_ovamta_stage1_cls_metrics(module):
    train_cls_metrics = build_ovamta_stage1_classification_metrics()
    val_cls_metrics = build_ovamta_stage1_classification_metrics()

    module.train_cls_f1 = train_cls_metrics["f1"]
    module.train_cls_precision = train_cls_metrics["precision"]
    module.train_cls_recall = train_cls_metrics["recall"]
    module.train_cls_auc = train_cls_metrics["auc"]

    module.val_cls_f1 = val_cls_metrics["f1"]
    module.val_cls_precision = val_cls_metrics["precision"]
    module.val_cls_recall = val_cls_metrics["recall"]
    module.val_cls_auc = val_cls_metrics["auc"]


class LitSegClsModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        aux_params = {
            "pooling": "avg",
            "dropout": 0.5,
            "activation": None,
            "classes": 2,
        }

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            aux_params=aux_params
        )

        cls_weights = get_binary_cls_weights()
        self.register_buffer("cls_weights", cls_weights)

        self.seg_criterion, self.cls_criterion = build_common_losses(self.cls_weights)

        _assign_binary_cls_metrics(self)
        _assign_multiclass_seg_metrics(self)

    def forward(self, x):
        seg_logits, cls_logits = self.model(x)
        return seg_logits, cls_logits

    def training_step(self, batch, batch_idx):
        images, seg_masks, cls_labels = batch

        seg_logits, cls_logits = self(images)

        seg_loss = self.seg_criterion(seg_logits, seg_masks)
        cls_loss = self.cls_criterion(cls_logits, cls_labels)

        cls_preds = torch.argmax(cls_logits, dim=1)
        cls_acc = (cls_preds == cls_labels).float().mean()

        seg_preds = torch.argmax(seg_logits, dim=1)

        train_seg_iou = self.train_seg_iou(seg_preds, seg_masks)

        cls_probs = torch.softmax(cls_logits, dim=1)[:, 1]
        train_cls_auc = self.train_cls_auc(cls_probs, cls_labels)

        loss = seg_loss + cls_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True)
        self.log("train_cls_acc", cls_acc, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train_seg_dice", self.train_seg_dice(seg_preds, seg_masks), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_f1", self.train_cls_f1(cls_preds, cls_labels), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_precision", self.train_cls_precision(cls_preds, cls_labels), on_step=False, on_epoch=True)
        self.log("train_cls_recall", self.train_cls_recall(cls_preds, cls_labels), on_step=False, on_epoch=True)

        self.log("train_seg_iou", train_seg_iou, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_auc", train_cls_auc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, seg_masks, cls_labels = batch

        seg_logits, cls_logits = self(images)

        seg_loss = self.seg_criterion(seg_logits, seg_masks)
        cls_loss = self.cls_criterion(cls_logits, cls_labels)

        cls_probs = torch.softmax(cls_logits, dim=1)[:, 1]

        seg_preds = torch.argmax(seg_logits, dim=1)
        val_seg_iou = self.val_seg_iou(seg_preds, seg_masks)

        loss = seg_loss + cls_loss

        cls_preds = torch.argmax(cls_logits, dim=1)
        cls_acc = (cls_preds == cls_labels).float().mean()

        self.val_cls_auc.update(cls_probs, cls_labels)
        self.val_cls_f1.update(cls_preds, cls_labels)
        self.val_cls_precision.update(cls_preds, cls_labels)
        self.val_cls_recall.update(cls_preds, cls_labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_loss", seg_loss, on_step=False, on_epoch=True)
        self.log("val_cls_loss", cls_loss, on_step=False, on_epoch=True)
        self.log("val_cls_acc", cls_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_dice", self.val_seg_dice(seg_preds, seg_masks), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_iou", val_seg_iou, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx == 0:
            pred_masks = torch.argmax(seg_logits, dim=1)

            num_images = min(3, images.shape[0])
            val_examples = []

            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(3, 1, 1)

            class_labels_mask = {
                0: "background",
                1: "class_1",
                2: "class_2",
            }

            for i in range(num_images):
                image_vis = images[i].detach().cpu() * std.cpu() + mean.cpu()
                image_vis = image_vis.clamp(0, 1)
                image_np = image_vis.permute(1, 2, 0).numpy()

                true_mask_np = seg_masks[i].detach().cpu().numpy().astype(np.uint8)
                pred_mask_np = pred_masks[i].detach().cpu().numpy().astype(np.uint8)

                val_examples.append(
                    wandb.Image(
                        image_np,
                        masks={
                            "ground_truth": {
                                "mask_data": true_mask_np,
                                "class_labels": class_labels_mask,
                            },
                            "prediction": {
                                "mask_data": pred_mask_np,
                                "class_labels": class_labels_mask,
                            },
                        },
                        caption=f"Sample {i} | true cls={cls_labels[i].item()} | pred cls={cls_preds[i].item()}"
                    )
                )

            self.logger.experiment.log({
                "val_examples": val_examples,
                "global_step": self.global_step
            })

        return loss

    def on_validation_epoch_end(self):
        self.log("val_cls_auc", self.val_cls_auc.compute(), prog_bar=True)
        self.log("val_cls_f1", self.val_cls_f1.compute(), prog_bar=True)
        self.log("val_cls_precision", self.val_cls_precision.compute())
        self.log("val_cls_recall", self.val_cls_recall.compute())

        self.val_cls_auc.reset()
        self.val_cls_f1.reset()
        self.val_cls_precision.reset()
        self.val_cls_recall.reset()

    def configure_optimizers(self):
        pg = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            pg,
            lr=self.hparams.lr,
            weight_decay=1e-4
        )

        return optimizer


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        n = target.size(0)
        smooth = 1e-5

        input = torch.softmax(input, dim=1)

        target = torch.nn.functional.one_hot(
            target,
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        input_flat = input.view(n, -1)
        target_flat = target.view(n, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / (
            input_flat.sum(1) + target_flat.sum(1) + smooth
        )

        loss = 1 - dice.mean()
        return loss


class LossNet(torch.nn.Module):
    def __init__(self, resize=True):
        super().__init__()

        vgg = torchvision.models.vgg16(pretrained=True).features

        self.blocks = torch.nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ])

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.resize = resize
        self.transform = torch.nn.functional.interpolate

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = self.transform(input, size=(256, 256), mode="bilinear", align_corners=False)
            target = self.transform(target, size=(256, 256), mode="bilinear", align_corners=False)

        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)

        return loss


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum


class LitACSNet(L.LightningModule):
    def __init__(self, lr=1e-3, seg_loss_mode="ce", gamma=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.model = MyModel()
        self.seg_loss_mode = seg_loss_mode
        self.gamma = gamma

        cls_weights = get_binary_cls_weights()
        self.register_buffer("cls_weights", cls_weights)

        self.seg_criterion, self.cls_criterion = build_common_losses(self.cls_weights)

        if self.seg_loss_mode == "ce_lossnet":
            self.lossnet = LossNet(resize=True)
        else:
            self.lossnet = None

        self.awl = AutomaticWeightedLoss(num=2)

        _assign_multiclass_seg_metrics(self)
        _assign_binary_cls_metrics(self)

    def forward(self, x):
        class_logits, seg_logits = self.model(x)
        return class_logits, seg_logits

    def compute_seg_loss(self, seg_logits, seg_masks):
        ce_seg_loss = self.seg_criterion(seg_logits, seg_masks)

        if self.seg_loss_mode == "ce":
            return ce_seg_loss, ce_seg_loss, None

        elif self.seg_loss_mode == "ce_lossnet":
            seg_probs = torch.softmax(seg_logits, dim=1)

            seg_masks_onehot = torch.nn.functional.one_hot(
                seg_masks,
                num_classes=3
            ).permute(0, 3, 1, 2).float()

            lossnet_loss = self.lossnet(seg_probs, seg_masks_onehot)
            seg_loss = ce_seg_loss + self.gamma * lossnet_loss

            return seg_loss, ce_seg_loss, lossnet_loss

        else:
            raise ValueError(
                f"Invalid value for 'seg_loss_mode': {self.seg_loss_mode}. "
                "Expected one of ['ce', 'ce_lossnet']."
            )

    def training_step(self, batch, batch_idx):
        images, seg_masks, cls_labels = batch

        class_logits, seg_logits = self(images)

        cls_labels = cls_labels.long()
        seg_masks = seg_masks.long()

        seg_loss, ce_seg_loss, lossnet_loss = self.compute_seg_loss(seg_logits, seg_masks)
        cls_loss = self.cls_criterion(class_logits, cls_labels)

        loss = self.awl(seg_loss, cls_loss)

        seg_probs = torch.softmax(seg_logits, dim=1)
        seg_preds = torch.argmax(seg_probs, dim=1)

        cls_probs = torch.softmax(class_logits, dim=1)[:, 1]
        cls_preds = torch.argmax(class_logits, dim=1)

        cls_acc = (cls_preds == cls_labels).float().mean()

        self.log("train_ce_seg_loss", ce_seg_loss, on_step=True, on_epoch=True)

        if lossnet_loss is not None:
            self.log("train_lossnet_loss", lossnet_loss, on_step=True, on_epoch=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True)

        self.log("train_cls_acc", cls_acc, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train_seg_dice", self.train_seg_dice(seg_preds, seg_masks), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_seg_iou", self.train_seg_iou(seg_preds, seg_masks), prog_bar=True, on_step=False, on_epoch=True)

        self.log("train_cls_f1", self.train_cls_f1(cls_preds, cls_labels), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_precision", self.train_cls_precision(cls_preds, cls_labels), on_step=False, on_epoch=True)
        self.log("train_cls_recall", self.train_cls_recall(cls_preds, cls_labels), on_step=False, on_epoch=True)
        self.log("train_cls_auc", self.train_cls_auc(cls_probs, cls_labels), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, seg_masks, cls_labels = batch

        class_logits, seg_logits = self(images)

        cls_labels = cls_labels.long()
        seg_masks = seg_masks.long()

        seg_loss, ce_seg_loss, lossnet_loss = self.compute_seg_loss(seg_logits, seg_masks)
        cls_loss = self.cls_criterion(class_logits, cls_labels)

        loss = self.awl(seg_loss, cls_loss)

        seg_probs = torch.softmax(seg_logits, dim=1)
        seg_preds = torch.argmax(seg_probs, dim=1)

        cls_probs = torch.softmax(class_logits, dim=1)[:, 1]
        cls_preds = torch.argmax(class_logits, dim=1)

        cls_acc = (cls_preds == cls_labels).float().mean()

        self.val_cls_auc.update(cls_probs, cls_labels)
        self.val_cls_f1.update(cls_preds, cls_labels)
        self.val_cls_precision.update(cls_preds, cls_labels)
        self.val_cls_recall.update(cls_preds, cls_labels)

        self.log("val_ce_seg_loss", ce_seg_loss, on_step=False, on_epoch=True)

        if lossnet_loss is not None:
            self.log("val_lossnet_loss", lossnet_loss, on_step=False, on_epoch=True)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_loss", seg_loss, on_step=False, on_epoch=True)
        self.log("val_cls_loss", cls_loss, on_step=False, on_epoch=True)

        self.log("val_cls_acc", cls_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_dice", self.val_seg_dice(seg_preds, seg_masks), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_iou", self.val_seg_iou(seg_preds, seg_masks), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_cls_auc", self.val_cls_auc.compute(), prog_bar=True)
        self.log("val_cls_f1", self.val_cls_f1.compute(), prog_bar=True)
        self.log("val_cls_precision", self.val_cls_precision.compute())
        self.log("val_cls_recall", self.val_cls_recall.compute())

        self.val_cls_auc.reset()
        self.val_cls_f1.reset()
        self.val_cls_precision.reset()
        self.val_cls_recall.reset()

    def configure_optimizers(self):
        pg = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            pg,
            lr=self.hparams.lr,
            weight_decay=1e-4
        )

        return optimizer


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))

    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


class LitOvaSeg(L.LightningModule):
    def __init__(self, lr=1e-4, cls_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["cls_weights"])

        self.model = TransRaUNet_CLF_xiaorong(training=True)

        if cls_weights is None:
            cls_weights = torch.tensor([0.0, 1.397, 0.779], dtype=torch.float)

        self.register_buffer("cls_weights", cls_weights)

        self.cls_ce = nn.CrossEntropyLoss(weight=self.cls_weights)
        self.cls_reg = SmoothL1Loss()

        _assign_ovamta_stage1_cls_metrics(self)
        _assign_binary_seg_metrics(self)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, images, gts, labels):
        out5, out4, out3, out2, cls_out, features = self.model(images)

        loss5 = structure_loss(out5, gts)
        loss4 = structure_loss(out4, gts)
        loss3 = structure_loss(out3, gts)
        loss2 = structure_loss(out2, gts)

        weight = self.cls_weights
        loss1 = self.cls_ce(cls_out, labels) + self.cls_reg(cls_out, labels, weight=weight)

        seg_loss = loss2 + loss3 + loss4 + loss5
        loss = loss1 + seg_loss

        return loss, loss1, loss2, loss3, loss4, loss5, cls_out, out5, out4, out3, out2

    def training_step(self, batch, batch_idx):
        images, gts, labels = batch

        loss, loss1, loss2, loss3, loss4, loss5, cls_out, out5, out4, out3, out2 = self.compute_loss(
            images, gts, labels
        )

        seg_loss = loss2 + loss3 + loss4 + loss5

        seg_logits = out5 + out4 + out3 + out2
        seg_preds = (seg_logits > 0).long()
        gts_int = gts.long()

        cls_preds = torch.argmax(cls_out, dim=1)
        cls_acc = (cls_preds == labels).float().mean()

        cls_probs = torch.softmax(cls_out, dim=1)

        train_seg_iou = self.train_seg_iou(seg_preds, gts_int)
        train_cls_auc = self.train_cls_auc(cls_probs, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cls_loss", loss1, on_step=True, on_epoch=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True)

        self.log("train_seg_loss2", loss2, on_step=True, on_epoch=True)
        self.log("train_seg_loss3", loss3, on_step=True, on_epoch=True)
        self.log("train_seg_loss4", loss4, on_step=True, on_epoch=True)
        self.log("train_seg_loss5", loss5, on_step=True, on_epoch=True)

        self.log("train_cls_acc", cls_acc, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train_seg_dice", self.train_seg_dice(seg_preds, gts_int), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_seg_iou", train_seg_iou, prog_bar=True, on_step=False, on_epoch=True)

        self.log("train_cls_f1", self.train_cls_f1(cls_preds, labels), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_precision", self.train_cls_precision(cls_preds, labels), on_step=False, on_epoch=True)
        self.log("train_cls_recall", self.train_cls_recall(cls_preds, labels), on_step=False, on_epoch=True)
        self.log("train_cls_auc", train_cls_auc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, gts, labels = batch

        loss, loss1, loss2, loss3, loss4, loss5, cls_out, out5, out4, out3, out2 = self.compute_loss(
            images, gts, labels
        )

        seg_loss = loss2 + loss3 + loss4 + loss5

        seg_logits = out5 + out4 + out3 + out2
        seg_preds = (seg_logits > 0).long()
        gts_int = gts.long()

        cls_preds = torch.argmax(cls_out, dim=1)
        cls_acc = (cls_preds == labels).float().mean()

        cls_probs = torch.softmax(cls_out, dim=1)

        self.val_cls_auc.update(cls_probs, labels)
        self.val_cls_f1.update(cls_preds, labels)
        self.val_cls_precision.update(cls_preds, labels)
        self.val_cls_recall.update(cls_preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_cls_loss", loss1, on_step=False, on_epoch=True)
        self.log("val_seg_loss", seg_loss, on_step=False, on_epoch=True)

        self.log("val_seg_loss2", loss2, on_step=False, on_epoch=True)
        self.log("val_seg_loss3", loss3, on_step=False, on_epoch=True)
        self.log("val_seg_loss4", loss4, on_step=False, on_epoch=True)
        self.log("val_seg_loss5", loss5, on_step=False, on_epoch=True)

        self.log("val_cls_acc", cls_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_dice", self.val_seg_dice(seg_preds, gts_int), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_iou", self.val_seg_iou(seg_preds, gts_int), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        pg = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            pg,
            lr=self.hparams.lr,
            weight_decay=1e-4
        )

        return optimizer

    def on_validation_epoch_end(self):
        self.log("val_cls_auc", self.val_cls_auc.compute(), prog_bar=True)
        self.log("val_cls_f1", self.val_cls_f1.compute(), prog_bar=True)
        self.log("val_cls_precision", self.val_cls_precision.compute())
        self.log("val_cls_recall", self.val_cls_recall.compute())

        self.val_cls_auc.reset()
        self.val_cls_f1.reset()
        self.val_cls_precision.reset()
        self.val_cls_recall.reset()


class LitOvaDiag(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = OvaDiagModel(training=True)

        cls_weights = get_binary_cls_weights()
        self.register_buffer("cls_weights", cls_weights)

        self.cls_ce = nn.CrossEntropyLoss(weight=self.cls_weights)
        self.cls_reg = SmoothL1Loss()

        _assign_binary_cls_metrics(self)
        _assign_binary_seg_metrics(self)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, images, gts, labels):
        out5, out4, out3, out2, cls_out, features = self.model(images)

        loss5 = structure_loss(out5, gts)
        loss4 = structure_loss(out4, gts)
        loss3 = structure_loss(out3, gts)
        loss2 = structure_loss(out2, gts)

        weight = self.cls_weights
        loss1 = self.cls_ce(cls_out, labels) + self.cls_reg(cls_out, labels, weight=weight)

        seg_loss = loss2 + loss3 + loss4 + loss5
        loss = loss1 + seg_loss

        return loss, loss1, loss2, loss3, loss4, loss5, cls_out, out5, out4, out3, out2

    def training_step(self, batch, batch_idx):
        images, gts, labels = batch

        loss, loss1, loss2, loss3, loss4, loss5, cls_out, out5, out4, out3, out2 = self.compute_loss(
            images, gts, labels
        )

        seg_loss = loss2 + loss3 + loss4 + loss5

        seg_logits = out5 + out4 + out3 + out2
        seg_preds = (seg_logits > 0).long()
        gts_int = gts.long()

        cls_preds = torch.argmax(cls_out, dim=1)
        cls_acc = (cls_preds == labels).float().mean()

        cls_probs = torch.softmax(cls_out, dim=1)[:, 1]

        train_seg_iou = self.train_seg_iou(seg_preds, gts_int)
        train_cls_auc = self.train_cls_auc(cls_probs, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_cls_loss", loss1, on_step=True, on_epoch=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True)

        self.log("train_seg_loss2", loss2, on_step=True, on_epoch=True)
        self.log("train_seg_loss3", loss3, on_step=True, on_epoch=True)
        self.log("train_seg_loss4", loss4, on_step=True, on_epoch=True)
        self.log("train_seg_loss5", loss5, on_step=True, on_epoch=True)

        self.log("train_cls_acc", cls_acc, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train_seg_dice", self.train_seg_dice(seg_preds, gts_int), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_seg_iou", train_seg_iou, prog_bar=True, on_step=False, on_epoch=True)

        self.log("train_cls_f1", self.train_cls_f1(cls_preds, labels), prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_cls_precision", self.train_cls_precision(cls_preds, labels), on_step=False, on_epoch=True)
        self.log("train_cls_recall", self.train_cls_recall(cls_preds, labels), on_step=False, on_epoch=True)
        self.log("train_cls_auc", train_cls_auc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, gts, labels = batch

        loss, loss1, loss2, loss3, loss4, loss5, cls_out, out5, out4, out3, out2 = self.compute_loss(
            images, gts, labels
        )

        seg_loss = loss2 + loss3 + loss4 + loss5

        seg_logits = out5 + out4 + out3 + out2
        seg_preds = (seg_logits > 0).long()
        gts_int = gts.long()

        cls_preds = torch.argmax(cls_out, dim=1)
        cls_acc = (cls_preds == labels).float().mean()

        cls_probs = torch.softmax(cls_out, dim=1)[:, 1]

        self.val_cls_auc.update(cls_probs, labels)
        self.val_cls_f1.update(cls_preds, labels)
        self.val_cls_precision.update(cls_preds, labels)
        self.val_cls_recall.update(cls_preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_cls_loss", loss1, on_step=False, on_epoch=True)
        self.log("val_seg_loss", seg_loss, on_step=False, on_epoch=True)

        self.log("val_seg_loss2", loss2, on_step=False, on_epoch=True)
        self.log("val_seg_loss3", loss3, on_step=False, on_epoch=True)
        self.log("val_seg_loss4", loss4, on_step=False, on_epoch=True)
        self.log("val_seg_loss5", loss5, on_step=False, on_epoch=True)

        self.log("val_cls_acc", cls_acc, prog_bar=True, on_step=False, on_epoch=True)

        self.log("val_seg_dice", self.val_seg_dice(seg_preds, gts_int), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_seg_iou", self.val_seg_iou(seg_preds, gts_int), prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_cls_auc", self.val_cls_auc.compute(), prog_bar=True)
        self.log("val_cls_f1", self.val_cls_f1.compute(), prog_bar=True)
        self.log("val_cls_precision", self.val_cls_precision.compute())
        self.log("val_cls_recall", self.val_cls_recall.compute())

        self.val_cls_auc.reset()
        self.val_cls_f1.reset()
        self.val_cls_precision.reset()
        self.val_cls_recall.reset()

    def configure_optimizers(self):
        pg = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            pg,
            lr=self.hparams.lr,
            weight_decay=1e-4
        )

        return optimizer