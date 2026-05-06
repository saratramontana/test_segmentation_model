import torch
from torch.utils.data import DataLoader

from training.datasets import (
    BaselineDataset,
    ACSNetMulticlassDataset,
    OvaSegWrapperDataset,
    OvaDiagWrapperDataset,
)

from training.lightning_modules import (
    LitSegClsModel,
    LitACSNet,
    LitOvaSeg,
    LitOvaDiag,
)


def add_ovamta_stage1_label(df):
    df = df.copy()

    if "ovamta_stage1_label" not in df.columns:
        df["ovamta_stage1_label"] = df["risk_class"].map({
            0: 1,
            1: 2,
        })

    return df


def compute_ovamta_stage1_weights(train_df):
    counts = train_df["ovamta_stage1_label"].value_counts().reindex([0, 1, 2], fill_value=0)

    weights = len(train_df) / (3 * counts)
    weights = weights.replace([float("inf")], 0.0)

    cls_weights = torch.tensor(weights.values, dtype=torch.float)

    return cls_weights


def build_dataloaders(model_name, train_df, val_df, batch_size):
    if model_name == "baseline":
        train_dataset = BaselineDataset(train_df)
        val_dataset = BaselineDataset(val_df)

    elif model_name == "acsnet":
        train_dataset = ACSNetMulticlassDataset(train_df)
        val_dataset = ACSNetMulticlassDataset(val_df)

    elif model_name == "ovamta_seg":
        train_df = add_ovamta_stage1_label(train_df)
        val_df = add_ovamta_stage1_label(val_df)

        train_dataset = OvaSegWrapperDataset(train_df)
        val_dataset = OvaSegWrapperDataset(val_df)

    elif model_name == "ovamta_diag":
        train_dataset = OvaDiagWrapperDataset(train_df)
        val_dataset = OvaDiagWrapperDataset(val_df)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def build_lightning_module(
    model_name,
    lr,
    train_df=None,
    seg_loss_mode="ce",
    gamma=0.2,
):
    if model_name == "baseline":
        return LitSegClsModel(lr=lr)

    elif model_name == "acsnet":
        return LitACSNet(
            lr=lr,
            seg_loss_mode=seg_loss_mode,
            gamma=gamma,
        )

    elif model_name == "ovamta_seg":
        if train_df is not None:
            train_df = add_ovamta_stage1_label(train_df)
            cls_weights = compute_ovamta_stage1_weights(train_df)
        else:
            cls_weights = None

        return LitOvaSeg(
            lr=lr,
            cls_weights=cls_weights,
        )

    elif model_name == "ovamta_diag":
        return LitOvaDiag(lr=lr)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")