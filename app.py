from src.neopolyp.model.model import NeoPolypModel
from src.neopolyp.dataset.coco_utils import *
from src.neopolyp.dataset.transforms import *
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr
import numpy as np
import torch
import os
import cv2

def mask2rgb(mask):
    color_dict = {0: torch.tensor([0, 0, 0]),
                  1: torch.tensor([1, 0, 0]),
                  2: torch.tensor([0, 1, 0])}
    output = torch.zeros((mask.shape[0], mask.shape[1], 3)).long()
    for k in color_dict.keys():
        output[mask.long() == k] = color_dict[k]
    return output.to(mask.device)

def change_input(img_input, choice):
    img, _, _, _ = dataset[choice]
    img_input = np.array(img)
    return img_input

def random_input(img_input):
    img, _, _, _ = dataset[np.random.randint(0, len(dataset))]
    img_input = np.array(img)
    return img_input

def aug(img):
    return A.Compose([
        A.Resize(224),
        A.Normalize(),
        ToTensorV2(),
    ])(image=img)['image']

def predict(img):
    H, W = img.shape[:2]
    img = aug(img).unsqueeze(0)
    pred = model(img).squeeze(0)
    argmax = torch.argmax(pred, 0)
    one_hot = mask2rgb(argmax).float().permute(2, 0, 1)
    mask2img = Resize((H, W), interpolation=InterpolationMode.NEAREST)(ToPILImage()(one_hot))
    return mask2img

def demo():
    # write a demo with gradio, input is image, output is image
    with gr.Blocks() as demo:
        gr.Markdown("# NeoPolyp Demo")
        with gr.Row():
            choice = gr.Slider(label="Image Index", minimum=0, maximum=len(dataset) - 1, step=1, value=0)
            img, _, _, _ = dataset[choice.value]

        with gr.Row():
            global img_input
            img_input = gr.Image(value=np.array(img))
            img_output = gr.Image()
            choice.change(change_input, inputs=[img_input, choice], outputs=[img_input])
        with gr.Row():
            random_btn = gr.Button("Random")
            pred_btn = gr.Button("Predict")

            pred_btn.click(predict, [img_input], outputs=img_output)
            random_btn.click(random_input, [img_input], outputs=img_input)

    demo.launch()
    
if __name__ == "__main__":
    root = "data/val2017"
    all_path = []
    for file in os.listdir(root):
        all_path.append(os.path.join(root, file))
        
    dataset = infer_build(mode="val")
    map_location = {'cuda:0':'cpu'}
    model = NeoPolypModel.load_from_checkpoint("./checkpoints/model/model.ckpt")
    demo()