import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = torch.tensor(np.frombuffer(row["image"], dtype=np.uint8).reshape(224,224)).float()/255 #frombuffer to convert bytes into integers from 0 to 255, then float because pytorch works with float
        image = image.unsqueeze(0)

        seg_mask = torch.tensor(np.frombuffer(row["gt_mask"], dtype=np.uint8).reshape(224,224)).float()/255
        seg_mask = (seg_mask > 0.5).float().unsqueeze(0) #First a boolean mask, then with float it becomes a binary mask and with unsqueeze I add a dimension

        cls_label = torch.tensor(1 if row["risk_class"]=="malignant" else 0)

        return image, seg_mask, cls_label