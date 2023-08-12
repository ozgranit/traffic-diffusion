import torch

from PIL import Image
from pathlib import Path
from prompts import prompt_getter
from diffusers import DiffusionPipeline


class ImageGenerator:
    def __init__(self):
        # Load both base & refiner

        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            variant="fp16",
            use_safetensors=True
        )

        base.to("cuda")
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float32,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

    def generate_image(self, original_image, prompt):
        # Define how many steps and what % of steps to be run on each expert (80/20) here
        n_steps = 40
        high_noise_frac = 0.8

        new_image = self.refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=original_image,
        ).images[0]

        return new_image

    def generate_variations(self, original_image, image_file_name, run_over_existing_images=False):

        image_dir = Path.cwd() / 'dataset_images' / image_file_name
        image_dir.mkdir(exist_ok=True)

        for transformation, prompt in prompt_getter.items():
            transformed_image = self.generate_image(original_image, prompt)

            transformed_image_path = image_dir / f'{transformation}.png'

            if transformed_image_path.exists() and not run_over_existing_images:
                continue

            transformed_image.save(transformed_image_path)

    def generate_all_images(self):

        original_images_path = Path.cwd() / 'kaggle_images'

        # Iterate over files and directories in the kaggle_images directory
        for item in original_images_path.iterdir():

            if item.is_file():
                # Load the image using PIL
                original_image = Image.open(item.resolve())
                self.generate_variations(original_image, item.name, run_over_existing_images=True)

            else:
                raise IsADirectoryError("Should only find images in the kaggle_images directory")


if __name__ == "__main__":
    image_gen = ImageGenerator()
    image_gen.generate_all_images()
