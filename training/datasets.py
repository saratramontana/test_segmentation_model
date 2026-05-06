import io
import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class BaselineDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(io.BytesIO(row["image"])).convert("RGB")
        image = self.img_transform(image)

        mask = Image.open(io.BytesIO(row["mask"])).convert("L")
        seg_mask = torch.tensor(np.array(mask), dtype=torch.long)

        cls_label = torch.tensor(row["risk_class"], dtype=torch.long)

        return image, seg_mask, cls_label


class ACSNetMulticlassDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.330, 0.330, 0.330],
                std=[0.204, 0.204, 0.204],
            ),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(io.BytesIO(row["image"])).convert("RGB")
        image = self.img_transform(image)

        mask = Image.open(io.BytesIO(row["mask"])).convert("L")
        mask = self.mask_transform(mask)
        seg_mask = torch.tensor(np.array(mask), dtype=torch.long)

        cls_label = torch.tensor(row["risk_class"], dtype=torch.long)

        return image, seg_mask, cls_label


class OvaSegWrapperDataset(Dataset):
    def __init__(self, df, trainsize=352):
        self.df = df.reset_index(drop=True)
        self.trainsize = trainsize

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize(
                (trainsize, trainsize),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(io.BytesIO(row["image"])).convert("RGB")
        gt = Image.open(io.BytesIO(row["mask"])).convert("L")

        image = self.img_transform(image)

        gt_np = np.array(gt)
        gt_bin = (gt_np > 0).astype(np.uint8)

        gt = Image.fromarray(gt_bin * 255)
        gt = self.gt_transform(gt)
        gt = (gt > 0).float()

        label = torch.tensor(row["ovamta_stage1_label"], dtype=torch.long)

        return image, gt, label


class OvaDiagWrapperDataset(Dataset):
    def __init__(self, df, trainsize=352):
        self.df = df.reset_index(drop=True)
        self.trainsize = trainsize

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize(
                (trainsize, trainsize),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(io.BytesIO(row["image"])).convert("RGB")
        mask = Image.open(io.BytesIO(row["mask"])).convert("L")

        mask_np = np.array(mask)
        mask_bin = (mask_np > 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        orig_w, orig_h = image.size

        if len(contours) == 0:
            x_min, y_min, x_max, y_max = 0, 0, orig_w, orig_h
        else:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            margin = 10

            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(orig_w, x + w + margin)
            y_max = min(orig_h, y + h + margin)

            if x_max <= x_min or y_max <= y_min:
                x_min, y_min, x_max, y_max = 0, 0, orig_w, orig_h

        image_patch = image.crop((x_min, y_min, x_max, y_max))
        mask_patch = Image.fromarray(mask_bin * 255).crop((x_min, y_min, x_max, y_max))

        image_patch = self.img_transform(image_patch)

        gt_patch = self.gt_transform(mask_patch)
        gt_patch = (gt_patch > 0).float()

        label = torch.tensor(row["risk_class"], dtype=torch.long)

        return image_patch, gt_patch, label