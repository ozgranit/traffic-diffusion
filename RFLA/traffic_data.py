import os

from load_images import process_image


class TrafficData:
    def __init__(self, attack_db: str, parent_dir: str = 'larger_images', image_inputs_dir: str = 'image_inputs',
                 image_annotations_dir: str = 'image_annotations',
                 mask_dir: str = r'image_masks', crop_size: int = 224):

        self.attack_db = attack_db
        self.crop_size = crop_size
        self.set_dirs(image_annotations_dir, image_inputs_dir, mask_dir, parent_dir)
        self.file_names, self.orig_imgs, self.cropped_imgs, self.cropped_resized_imgs, \
            self.labels, self.bbx, self.masks_cropped = process_image(
            self.image_inputs_dir,  # kaggle_images',
            # /workspace/traffic-`diffusion/
            self.image_annotations_dir,
            self.attack_db, crop_size=crop_size[0], mask_folder=self.mask_dir)  # /workspace/traffic-diffusion/


    def set_dirs(self, image_annotations_dir, image_inputs_dir, mask_dir, parent_dir):
        if parent_dir:
            self.image_inputs_dir = os.path.join(parent_dir, image_inputs_dir)
            self.image_annotations_dir = os.path.join(parent_dir, image_annotations_dir)
            self.mask_dir = os.path.join(parent_dir, mask_dir)
        else:
            self.image_inputs_dir = image_inputs_dir
            self.image_annotations_dir = image_annotations_dir
            self.mask_dir = mask_dir
