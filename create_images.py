import torch
import gc
from PIL import Image
from pathlib import Path
from prompts import prompt_getter
from diffusers import StableDiffusionDepth2ImgPipeline


class ImageGenerator:
    def __init__(self):
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        ).to("cuda")

    def generate_image(self, original_image, prompt):
        torch.cuda.empty_cache()
        gc.collect()

        strength = 0.45
        new_image =self.pipe(prompt="A photo, " + prompt, image=original_image,
                             negative_prompt="Cartoon, Disfigured, blurry, unrealistic", strength=strength).images[0]

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
                original_image = Image.open(item.resolve()).convert("RGB")
                self.generate_variations(original_image, item.name, run_over_existing_images=True)

            else:
                raise IsADirectoryError("Should only find images in the kaggle_images directory")


if __name__ == "__main__":
    image_gen = ImageGenerator()
    image_gen.generate_all_images()
