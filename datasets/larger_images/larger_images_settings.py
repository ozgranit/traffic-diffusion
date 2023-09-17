import os.path

# Larger images
LARGER_IMAGES = 'larger_images'
LARGER_IMAGES_DIR = rf'datasets/{LARGER_IMAGES}'
LARGER_IMAGES_INPUTS = os.path.join(LARGER_IMAGES_DIR, 'image_inputs')
LARGER_IMAGES_ANNOTATIONS = os.path.join(LARGER_IMAGES_DIR, 'image_annotations')
LARGER_IMAGES_MASKS = os.path.join(LARGER_IMAGES_DIR, 'image_masks')

LARGER_IMAGES_DIFFUSION = r'datasets/larger_images/image_outputs'    #TODO: move ouput image to new larger images folder in datasets

