from src.segmentation.model.model import NeoPolypModel
from src.segmentation.dataset.coco_utils import *
# from src.neopolyp.dataset.transforms import *
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr
import numpy as np
import torch
import os
import cv2

device = torch.device("cpu")
model = NeoPolypModel.load_from_checkpoint("./checkpoints/model/model.ckpt")
model.eval()

def mask2rgb(mask):
    color_dict = {
    0: torch.tensor([0, 0, 0]),
    1: torch.tensor([1, 0, 0]),
    2: torch.tensor([0, 1, 0]),
}
    output = torch.zeros((mask.shape[0], mask.shape[1], 3)).long()
    for k in color_dict.keys():
        output[mask.long() == k] = color_dict[k]
    return output.to(mask.device)


def visualize_instance_seg_mask(mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    labels = np.unique(mask)
    label2color = {label: (random.randint(0, 1), random.randint(0, 255), random.randint(0, 255)) for label in labels}
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        image[i, j, :] = label2color[mask[i, j]]
    image = image / 255
    return image

def predict(img):
    img = torch.permute(torch.tensor(img), ( 2, 0, 1))
    H, W = img.shape[1:]
    img = img.unsqueeze(0)
    pred = model(img.float()).squeeze(0)
    argmax = torch.argmax(pred, 0)
    one_hot = torch.permute(mask2rgb(argmax).float(), (2, 0, 1))
    mask2img =Resize(256, interpolation=InterpolationMode.NEAREST)(ToPILImage()(one_hot))
    # mask2img = visualize_instance_seg_mask(mask2img)
    return mask2img

dataset = infer_build(mode="val")
img, mask = dataset[0]

demo = gr.Interface(
    predict, 
    inputs=[gr.Image(np.array(torch.permute(img, (1, 2, 0)), dtype=np.uint8))], 
    outputs="image",
    title="Image Segmentation Demo",
    description = "Please upload an image to see segmentation capabilities of this model"
)

demo.launch()