import torch
import torch.nn as nn
import pytorch_lightning as pl
from .unet import UNet
from .resunet import Resnet50Unet
from .loss import dice_score, DiceLoss
from .psp_net import PSPNet
from .seg_net import SegNet

class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, name: str = "resunet"):
        super().__init__()
        if name == "resunet":
            self.model = Resnet50Unet(n_classes=21)
        if name == "pspnet":
            self.model = PSPNet(num_classes=21)
        if name == "segnet":
            self.model = SegNet(num_classes=21)
        else:
            self.model = UNet(in_channels=3, attention=True, recurrent=False)
        self.lr = lr
        self.dice_loss = DiceLoss()
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, x):
        return self.model(x)

    def _forward(self, batch, batch_idx, name="train"):
        image, mask = batch[0].float(), batch[1].long()
        logits = self(image)
        loss = self.entropy_loss(logits, mask)
        # d_score = dice_score(logits, mask)
        acc = (logits.argmax(dim=1) == mask).float().mean()
        self.log_dict(
            {
                f"{name}_loss": loss,
                # f"{name}_dice_score": d_score,
                f"{name}_acc": acc
            },
            on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,

        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.config.pct_start
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]