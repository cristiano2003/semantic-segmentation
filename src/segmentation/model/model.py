import torch
import torch.nn as nn
import pytorch_lightning as pl
from .unet import UNet
from .resunet import Resnet50Unet
from .deeplabv3_plus import DeepLab
from .segnet import *
from .loss import DiceLoss

class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, name: str = "segresnet", num_classes = 6):
        super().__init__()
        if name == "resunet":
            self.model = Resnet50Unet(n_classes=num_classes)
        if name == "deeplabv3plus":
            self.model = DeepLab(num_classes=num_classes)
        if name == "segnet":
            self.model = SegResNet(num_classes=num_classes)
        else:
            self.model = UNet(in_channels=3, num_classes= num_classes, attention=True, recurrent=False)
        self.lr = lr
        self.dice_loss = DiceLoss()
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, x):
        return self.model(x)

    def _forward(self, batch, batch_idx, name="train"):
        image, mask = batch[0].float(), batch[1].long()
        logits = self(image)
        logits_d = logits.clone()
        mask_d = mask.clone()
        loss = self.entropy_loss(logits, mask)
        d_loss = self.dice_loss(logits_d, mask_d)
        acc = (logits.argmax(dim=1) == mask).float().mean()
        self.log_dict(
            {
                f"{name}_loss": loss,
                f"{name}_dice_loss": d_loss,
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
            lr=self.lr
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    
    model = NeoPolypModel()
    print(summary(model, input_size=(3, 256, 256), device="cpu"))