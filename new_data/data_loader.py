import pandas as pd


PROJECT_ROOT = "/Users/saratramontana/Documents/GitHub/test_segmentation_model"

TRAIN_PATH = f"{PROJECT_ROOT}/new_data/round_1_dataset_subset/train_fold_0_seed_0.parquet"
VAL_PATH = f"{PROJECT_ROOT}/new_data/round_1_dataset_subset/val_fold_0_seed_0.parquet"


train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)


