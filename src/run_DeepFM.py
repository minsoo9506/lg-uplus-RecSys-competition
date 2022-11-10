import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data_load.data_DeepFM import (
    DeepFMMakeBaselineData,
    DeepFMTrainTestSplit,
    DeepFMDataset,
)
from lit_model.DeepFM_lit_model import DeepFMLitModel
from model.DeepFM import DeepFM


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", default="DeepFM")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=16,
        help="embedding dimensions (default: 16)",
    )
    parser.add_argument(
        "--mlp_dims",
        default=[16, 16],
        help="mlp hidden layers' dimensions (default: [16, 16])",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 3)"
    )
    parser.add_argument("--cuda", type=int, default=0, help="0 for cpu -1 for all gpu")
    config = parser.parse_args()
    if config.cuda == 0 or torch.cuda.is_available() is False:
        config.cuda = 0

    return config


def main(config):
    history = pd.read_csv("../data/history_data.csv")
    meta = pd.read_csv("../data/meta_data.csv")
    profile = pd.read_csv("../data/profile_data.csv")
    data, meta_use, profile_use = DeepFMMakeBaselineData(history, meta, profile)
    unique_item = data["album_id"].unique()

    # split train, valid
    TEST_RATIO = 0.2
    split_train_valid = DeepFMTrainTestSplit(
        data=data, test_size=TEST_RATIO, random_seed=0
    )
    train, valid = split_train_valid.split()

    # train_dataset with negative sampling
    train_dataset = DeepFMDataset(
        data=train,
        is_train=True,
        DeepFMTrainTestSplit=split_train_valid,
        unique_item=unique_item,
        neg_ratio=1,  # neg_ratio -> config에 추가하기
        meta_use=meta_use,
        profile_use=profile_use,
    )
    valid_dataset = DeepFMDataset(
        valid, is_train=False, DeepFMTrainTestSplit=split_train_valid
    )

    # loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    # model
    DeepFM_model = DeepFM(
        field_dims=train_dataset.field_dims,
        embed_dim=config.embed_dim,
        mlp_dims=config.mlp_dims,
        dropout=0.2,
    )

    DeepFM_lit_model = DeepFMLitModel(DeepFM_model, config)

    # trainer
    # logger = pl.loggers.WandbLogger()
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="validation/loss", mode="min", patience=20
    )
    trainer = pl.Trainer(
        # logger=logger,
        log_every_n_steps=10,  # set the logging frequency
        gpus=config.cuda,  # use all GPUs
        max_epochs=config.epochs,  # number of epochs
        deterministic=True,  # keep it deterministic
        callbacks=[early_stopping_callback],
    )

    # fit the model
    trainer.fit(DeepFM_lit_model, train_loader, valid_loader)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
