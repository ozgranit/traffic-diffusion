import cv2
import numpy as np
import os
from plot_images import plot_image, plot_2_images_in_a_row
from prompts import prompt_getter
from thesis.create_images_api_inpainting import load_images
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image  #, make_image_grid
import torch
import sys
from seed import *

INPUT_FOLDER = r'datasets/imags_with_shadow/input_large_src_with_shadow' #'api_input/'
# OUTPUT_FOLDER = r'datasets/imags_with_shadow/output_api_inpainting_xl'   #'api_output/'
OUTPUT_FOLDER = r'datasets/imags_with_shadow/inpainting_xl_out'   #'api_output/'



def load_inpainting_models():
    base = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    return base, refiner

def inpainting_xl(base, refiner, init_image, mask_image, prompt):
    num_inference_steps = 75
    high_noise_frac = 0.7

    image = base(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        denoising_start=high_noise_frac,
    ).images[0]

    array_img = np.array(image)
    dest_img = os.path.join(OUTPUT_FOLDER, 'final.png')
    cv2.imwrite(dest_img, array_img)
    print(f"dest_img: {dest_img}")
    # plot_2_images_in_a_row(init_image, mask_image, "init_img", "mask_img")
    # plot_image(image.resize((512, 512)), 'gen')
    # make_image_grid([init_image, mask_image, image.resize((512, 512))], rows=1, cols=3)
    print("Finished!")


def generate_inpainting_locally(original_images):

    img_path = r'datasets/imags_with_shadow/input_large_src_with_shadow/road_1_with_attack.jpg'
    mask_path = r'datasets/imags_with_shadow/input_large_src_mask_attack/road_1.jpg'
    img = load_image(img_path).convert("RGB")
    mask = load_image(mask_path)
    # blurred_mask = pipeline.mask_processor.blur(mask, blur_factor=33)
    # img = cv2.imread(img_path)
    # mask = cv2.imread(mask_path)

    base, refiner = load_inpainting_models()
    blurred_mask  = base.mask_processor.blur(mask, blur_factor=33)
    original_images['road_1'] = [img, blurred_mask]

    for index, (filename, images) in enumerate(original_images.items()):
        image, mask = images[0], images[1]
        for prompt_desc, cur_prompt in prompt_getter.items():
            inpainting_xl(base, refiner, image, mask, cur_prompt)


def create_inpainting_imgs_locally():
    images_to_filter = []
    original_images = load_images(INPUT_FOLDER, images_to_filter)
    generate_inpainting_locally(original_images)

if __name__ == "__main__":
    create_inpainting_imgs_locally()