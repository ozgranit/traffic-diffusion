from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"
path_to_image = r'/home/mansour/ozgranit/traffic-diffusion/kaggle_image/road96.png'  # Replace with the correct path to your image file

# Load the image using PIL
image = Image.open(path_to_image)
# Resize the image to 128x128
desired_size = (128, 128)
image_resized = image.resize(desired_size, Image.ANTIALIAS)

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]



if __name__ == "__main__":
    image.save('/home/mansour/ozgranit/traffic-diffusion/kaggle_image/new.png')
