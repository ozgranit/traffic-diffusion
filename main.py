from attacks.RFLA import our_pos_reflect_attack_thesis
from consistency_check.diffusion_consistency_generation_check import diffusion_consistency_generation_check
import torch
import numpy as np
import random

# Set seed for PyTorch
# Check if CUDA is available and set seed for CUDA
from seed import *
from thesis.create_images_inpainting_locally import create_inpainting_imgs_locally

if __name__ == "__main__":
    our_pos_reflect_attack_thesis.main()

    # diffusion_consistency_generation_check()

    # create_inpainting_imgs_locally()