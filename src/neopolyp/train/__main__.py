from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize
from ..model.model import NeoPolypModel
from ..dataset.coco_utils import *
from ..dataset.data_utils import *
from ..dataset.data import *
import torch
import wandb
import pytorch_lightning as pl
import argparse
import random
import os

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # PARSERs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m', type=str, default='resunet',
        help='model name')
    parser.add_argument(
        '--coco_path', '-d', type=str, default='data',
        help='model name')
    parser.add_argument(
        '--max_epochs', '-me', type=int, default=200,
                        help='max epoch')
    parser.add_argument(
        '--batch_size', '-bs', type=int, default=25,
                        help='batch size')
    parser.add_argument(
        '--lr', '-l', type=float, default=1e-4,
        help='learning rate')
    parser.add_argument(
        '--num_workers', '-nw', type=int, default=2,
        help='number of workers')
    parser.add_argument(
        '--split_ratio', '-sr', type=float, default=0.9,
        help='split ratio')
    parser.add_argument(
        '--accumulate_grad_batches', '-agb', type=int, default=1,
        help='accumulate_grad_batches')
    parser.add_argument(
        '--seed', '-s', type=int, default=42,
        help='seed')
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument(
        '--wandb', '-w', default=False, action='store_true',
        help='use wandb or not')
    parser.add_argument(
        '--wandb_key', '-wk', type=str,
        help='wandb API key')

    args = parser.parse_args()

    # SEED
    pl.seed_everything(args.seed, workers=True)

    # WANDB (OPTIONAL)
    if args.wandb:
        wandb.login(key=args.wandb_key)  # API KEY
        name = f"{args.model}-{args.max_epochs}-{args.batch_size}-{args.lr}"
        logger = WandbLogger(project="semantic-segmentation",
                             name=name,
                             log_model="all")
    else:
        logger = None

    dataset = build("val", args=args)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # MODEL
    model = NeoPolypModel(lr=args.lr, name=args.model)

    # CALLBACK
    root_path = os.path.join(os.getcwd(), "checkpoints")
    ckpt_path = os.path.join(os.path.join(root_path, "model/"))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_path,
        filename="model",
        save_top_k=1,
        mode="min"
    )  # save top 2 epochs with the highest val_dice_score
    lr_callback = LearningRateMonitor("step")

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=15,
        verbose=True,
        mode="max"
    )

    # TRAINER
    trainer = pl.Trainer(
        default_root_dir=root_path,
        logger=logger,
        callbacks=[
            ckpt_callback, lr_callback, early_stop_callback
        ],
        gradient_clip_val=1.0,
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
        deterministic=False,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    # FIT MODEL
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
    
    