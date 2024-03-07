import torch
from diffusers import StableDiffusionXLImg2ImgPipeline


def load_stable_diffusion_xl():
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16
        # "stabilityai/'stable-diffusion-xl-base-1.0'", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    return pipe

