# Kaggle images
import os

KAGGLE_IMAGES = 'kaggle'
KAGGLE_IMAGES_DIR = rf'datasets/{KAGGLE_IMAGES}'
KAGGLE_IMAGES_INPUTS = os.path.join(KAGGLE_IMAGES_DIR, 'kaggle_images')
KAGGLE_IMAGES_ANNOTATIONS = os.path.join(KAGGLE_IMAGES_DIR, 'kaggle_annotations')
# KAGGLE_IMAGES_MASKS = os.path.join(KAGGLE_IMAGES_DIR, 'image_masks')

KAGGLE_IMAGES_DIFFUSION = os.path.join(KAGGLE_IMAGES_DIR, 'kaggle_diffusion_generated_images')
