from .transforms import *
from .data_utils import *
from torch.utils.data import random_split
from .coco_utils import get_coco_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

def build_transforms(is_train, crop_size,mode="baseline"):
    transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomGamma(gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
                A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(),
                        A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
                A.CoarseDropout(p=0.2, max_height=35, max_width=35, fill_value=255),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
                A.RandomShadow(p=0.1),
                A.ShiftScaleRotate(p=0.45, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.15),
                A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
                A.Normalize(),
                ToTensorV2(),
            ])
    return transform

def get_coco(root,batch_size=16, image_size=256,mode="custom1",num_workers=4):
    dataset=get_coco_dataset(root, "val", build_transforms(True, image_size,mode))
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    
    train_loader = get_dataloader_train(train_dataset, batch_size,num_workers)
    val_loader = get_dataloader_val(val_dataset,num_workers)
    return train_loader, val_loader


