import dataclasses

from datasets.larger_images.larger_images_settings import LARGER_IMAGES_INPUTS, \
    LARGER_IMAGES_ANNOTATIONS, \
    LARGER_IMAGES_MASKS,\
    LARGER_IMAGES_DIFFUSION

from datasets.kaggle.kaggle_images_settings import KAGGLE_IMAGES_INPUTS,\
    KAGGLE_IMAGES_ANNOTATIONS,\
    KAGGLE_IMAGES_DIFFUSION

class InputData:
    def __init__(self, data: str = "larger_images"):
        """
        data can be one of the following:
        1. larger_images
        2. kaggle
        """
        if data.lower() == "larger_images":
            self.input_name = data
            self.input_folder = LARGER_IMAGES_INPUTS
            self.annotations_folder = LARGER_IMAGES_ANNOTATIONS
            self.mask_folder = LARGER_IMAGES_MASKS
            self.diffusion_images = LARGER_IMAGES_DIFFUSION

        elif data.lower() == 'kaggle':
            self.input_folder = KAGGLE_IMAGES_INPUTS
            self.annotations_folder = KAGGLE_IMAGES_ANNOTATIONS
            self.mask_folder = None
            self.diffusion_images = KAGGLE_IMAGES_DIFFUSION


