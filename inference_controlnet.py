import re
import os
import json
import random
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms.functional import hflip
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL

from evaluation3 import evaluate
from data.helpers import normalize_sketch, increase_line_width
from data.prepare_datasets_for_augs2s import preprocess_video_sketch_for_validation

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_mptracking_img(path, resize):
    inp = Image.open(path)
    inp = transforms.ToTensor()(inp)
    inp = 1 - inp
    inp = inp[:,52:52+1028, 37:37+1028].clone()
    inp = normalize_sketch(inp)
    inp = resize(inp)
    inp = increase_line_width(inp.unsqueeze(0), kernel_size=args.line_width - 2).squeeze(0)
    if 'real' not in args.root:
        inp = hflip(inp)
    return inp


def inference_controlnet(args):

    os.makedirs(args.save_dir, exist_ok=True)
    resize = transforms.Resize((args.resolution, args.resolution))

    json.dump(vars(args), open(os.path.join(args.save_dir, "args.json"), "w"), indent=4)

    controlnet = ControlNetModel.from_pretrained(args.ckpt, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True
        )
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_model_cpu_offload()
    pipeline.to('cuda')

    pipeline.set_progress_bar_config(disable=True)
    
    sketch_dict = json.load(open(args.sketch_dict))

    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()

    file_list = os.listdir(args.root)
    
    random.seed(10)

    if args.num_samples is not None:
        file_list = random.sample(file_list, args.num_samples)

    os.makedirs(os.path.join(args.save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "mp"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "mp_gen"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "aug"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "aug_gen"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "combined"), exist_ok=True)

    for f in tqdm(file_list):

        pattern = r'^([^0-9]+)(\d+)\.png'
        match = re.match(pattern, f)
        category = match.group(1).replace(" ", "")
        idx = match.group(2)


        if os.path.exists(os.path.join(args.save_dir, "mp_gen", category, f)):
            continue

        aug_tensor, gt_img, prompt, augmented_sketch = preprocess_video_sketch_for_validation(category, idx, sketch_dict, args)
        mp_tensor = get_mptracking_img(os.path.join(args.root, f), resize)


        if not os.path.exists(os.path.join(args.save_dir, "mp_gen", category)):
            os.makedirs(os.path.join(args.save_dir, "mp_gen", category))
            os.makedirs(os.path.join(args.save_dir, "mp", category))
            os.makedirs(os.path.join(args.save_dir, "gt", category))
            os.makedirs(os.path.join(args.save_dir, "aug_gen", category))
            os.makedirs(os.path.join(args.save_dir, "aug", category))
            os.makedirs(os.path.join(args.save_dir, "combined", category))

        if args.no_prompt:
            prompt = ''

        aug_gen = pipeline(
            prompt=prompt, image=aug_tensor.unsqueeze(0), height=256, width=256, num_inference_steps=25
        ).images[0]

        mp_gen = pipeline(
            prompt=prompt, image=mp_tensor.unsqueeze(0), height=256, width=256, num_inference_steps=25
        ).images[0]

        aug_img = topil(aug_tensor)
        mp_img = topil(mp_tensor)

        aug_img.save(os.path.join(args.save_dir, "aug", category, f))
        mp_img.save(os.path.join(args.save_dir, "mp", category, f))
        aug_gen.save(os.path.join(args.save_dir, "aug_gen", category, f))
        mp_gen.save(os.path.join(args.save_dir, "aug_gen", category, f))
        gt_img.save(os.path.join(args.save_dir, "gt", category, f))
        image_grid([gt_img, aug_img, aug_gen, mp_img, mp_gen], 1,5).save(os.path.join(args.save_dir, "combined", category, f))
    
    if not args.no_eval:
        evaluate(args.root, save_path=os.path.join(args.save_dir, "stats.jsonl"), metrics=['ssim', 'lpips', 'psnr', 'chamfer_dist_points', 'chamfer_dist_images', 'clip'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--root", type=str)
    parser.add_argument("--sketch_dict", type=str, default="datasets/sketches_full.json")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--inverse_flow", action="store_true")
    parser.add_argument("--inverse_sketch", action="store_true")
    parser.add_argument("--line_width", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--prompt_format", type=str, default="a black and white sketch of a {}")
    parser.add_argument("--no_prompt", action='store_true')
    parser.add_argument("--no_eval", action='store_true')
    args = parser.parse_args()

    print(vars(args))
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    args.no_augmentation = False

    inference_controlnet(args)