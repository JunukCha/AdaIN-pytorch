import os
import argparse
from PIL import Image

import torch
import torch.optim as optim
import torchvision

from lib.datasets.data import get_dataloader
from lib.models.styletransfer import get_styletransfer
from lib.utils.img import get_demo_transform, unnormalize
from lib.utils.loss import calc_content_loss, calc_style_loss

def main(args):
    st_net = get_styletransfer()
    st_net.cuda()
    st_net.load_state_dict(torch.load(args.ckpt_path))
    st_net.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    content_image = Image.open(args.content_image).convert("RGB")
    style_image = Image.open(args.style_image).convert("RGB")

    demo_transform = get_demo_transform()
    content_image = demo_transform(content_image).unsqueeze(0)
    style_image = demo_transform(style_image).unsqueeze(0)
    output = st_net.generate(content_image, style_image)
    
    output_folder = "demo_out"
    os.makedirs(output_folder, exist_ok=True)

    content_image = unnormalize(content_image, mean, std)
    style_image = unnormalize(style_image, mean, std)
    output = unnormalize(output, mean, std)

    content_list = [content_image]
    style_list = [style_image]
    output_list = [output]

    combined_tensor = torch.stack(content_list + style_list + output_list, dim=0)
    save_path = f"demo_out/composite.png"
    torchvision.utils.save_image(combined_tensor, save_path)
    save_path = f"demo_out/out.png"
    torchvision.utils.save_image(output, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--content_image', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style_image', type=str, required=True, help='Path to the style image')
    args = parser.parse_args()
    main(args)