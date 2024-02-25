from .transforms import *
from .data_utils import *
from torch.utils.data import random_split
from .coco_utils import get_coco_dataset

def build_transforms(is_train, size, crop_size,mode="baseline"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
    ignore_value = 255
    transforms=[]
    min_scale=1
    max_scale=1
    if is_train:
        min_scale=0.5
        max_scale=2
    transforms.append(RandomResize(int(min_scale*size),int(max_scale*size)))
    if is_train:
        if mode=="baseline":
            pass
        elif mode=="randaug":
            transforms.append(RandAugment(2,1/3,prob=1.0,fill=fill,ignore_value=ignore_value))
        elif mode=="custom1":
            transforms.append(ColorJitter(0.5,0.5,(0.5,2),0.05))
            transforms.append(AddNoise(10))
            transforms.append(RandomRotation((-10,10), mean=fill, ignore_value=0))
        else:
            raise NotImplementedError()
        transforms.append(
        RandomCrop(
            crop_size,crop_size,
            fill,
            ignore_value,
            random_pad=is_train
        ))
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())
    transforms.append(Normalize(
        mean,
        std
    ))
    return Compose(transforms)

def get_coco(root,batch_size=16,val_size=513,train_size=481,mode="baseline",num_workers=4):
    dataset=get_coco_dataset(root, "train", build_transforms(True, val_size, train_size,mode))
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
    
    train_loader = get_dataloader_train(train_dataset, batch_size,num_workers)
    val_loader = get_dataloader_val(val_dataset,num_workers)
    return train_loader, val_loader


