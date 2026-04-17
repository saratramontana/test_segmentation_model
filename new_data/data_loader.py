

from torch.utils.data import DataLoader
import pandas as pd
from new_data.dataset import MyDataset

train_df = pd.read_parquet("/Users/saratramontana/Documents/test_segmentation_model/new_data/round_1_dataset_subset/train_fold_0_seed_0.parquet")
val_df = pd.read_parquet("/Users/saratramontana/Documents/test_segmentation_model/new_data/round_1_dataset_subset/val_fold_0_seed_0.parquet")

train_dataset = MyDataset(train_df)
val_dataset = MyDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


