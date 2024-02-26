import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test Kanji')
    
    parser.add_argument('--model_path', type=str, default='path_to_saved_model')
    
    parser.add_argument('--prompt', type=str, default='abandon')
    
    parser.add_argument('--num_images', type=int, default=10)
    
    parser.add_argument('--output_path', type=str, default='outputs/')
    
    parser.add_argument('--checkpoint', action=int, default=0)
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    if args.checkpoint:
        model_path = args.model_path
        unet = UNet2DConditionModel.from_pretrained(model_path + f"/checkpoint-{args.checkpoint}/unet", torch_dtype=torch.float16)

        pipe = StableDiffusionPipeline.from_pretrained("<initial model>", unet=unet, torch_dtype=torch.float16)
        pipe.to("cuda")

        for i in range(args.num_images):
            image = pipe(prompt=args.prompt).images[0]
            image.save(f"{args.prompt}_{i}.png")
    
    else:
        model_path = args.model_path
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.to("cuda")

        for i in range(args.num_images):
            image = pipe(prompt=args.prompt).images[0]
            image.save(f"{args.prompt}_{i}.png")


if __name__ == '__main__':
    main()