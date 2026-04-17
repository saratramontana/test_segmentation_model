from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.dataset import MyDataset
import pandas as pd

df=pd.read_parquet("/Users/saratramontana/Documents/test_segmentation_model/data/fake_train_fold_0_seed_0.parquet")


train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = MyDataset(train_df)
val_dataset = MyDataset(val_df)
test_dataset = MyDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)