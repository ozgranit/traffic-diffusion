import os.path

LARGER_IMAGES_DIR = r'datasets/larger_images'
LARGER_IMAGES_INPUTS = os.path.join(LARGER_IMAGES_DIR, 'image_inputs')
LARGER_IMAGES_ANNOTATIONS = os.path.join(LARGER_IMAGES_DIR, 'image_annotations')
LARGER_IMAGES_MASKS = os.path.join(LARGER_IMAGES_DIR, 'image_masks')

LARGER_IMAGES_DIFFUSION = r'larger_images/image_outputs'    #TODO: move ouput image to new larger images folder in datasets
