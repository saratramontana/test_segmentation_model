import io
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.img_transform = transform_img

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


