import copy
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data import random_split
import os

from pycocotools import mask as coco_mask

from .transforms import *

class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno

def build_transforms(is_train, mode="baseline"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
  
    transforms=[]
    
    transforms.append(RandomResize(256))
    
    if is_train:
        if mode=="baseline":
            pass
        elif mode=="randaug":
            transforms.append(RandAugment(2,1/3,prob=1.0,fill=fill,ignore_value=255))
        elif mode=="custom1":
            transforms.append(ColorJitter(0.5,0.5,(0.5,2),0.05))
            transforms.append(AddNoise(10))
            transforms.append(RandomRotation((-10,10), mean=fill, ignore_value=0))
        else:
            raise NotImplementedError()
        
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())
    # transforms.append(Normalize(
    #     mean,
    #     std
    # ))
    return Compose(transforms)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            # target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target



def build(args):
    root = args.coco_path
    PATHS =  ("val2017", os.path.join("annotations", "instances_val2017.json"))
      
    CAT_LIST = [i for i in range(1, 93)]
    
    transforms = Compose([
    # FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        ConvertCocoPolysToMask(),
        build_transforms(True, mode="randaug")
    ])

    img_folder, ann_file = PATHS
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
    train, val = random_split(dataset, [0.9, 0.1])

    return train, val

def infer_build():
    root = "data"
    PATHS =  ("val2017", os.path.join("annotations", "instances_val2017.json"))
      
    CAT_LIST = [i for i in range(1, 93)]

    transforms = Compose([
       # FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        ConvertCocoPolysToMask(),
        build_transforms(True)
    ])

    img_folder, ann_file = PATHS
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
    train, val = random_split(dataset, [0.9, 0.1])

    return train, val