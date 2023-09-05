import os
from typing import List

import cv2
import numpy as np
from PIL.Image import Image

from settings import GENERATED_IMAGES_TYPES_TRAIN, GENERATED_IMAGES_TYPES_TEST, ATTACK_TYPE_A, ATTACK_TYPE_B
from img_utils import crop_image

DIFFUSION_IMAGES_PATH = r'larger_images/image_outputs'
class DiffusionImages:
    def __init__(self, img_name, img_bbx, size, experiment_dir):
        # Loading generated images
        self.generated_dir_path: str = f'{DIFFUSION_IMAGES_PATH}/{img_name}'  # generated_images/road53'
        self.generated_imgs_train_cropped, self.generated_imgs_train_cropped_names = self.load_generated_augmentations_by_type(
            GENERATED_IMAGES_TYPES_TRAIN, self.generated_dir_path, img_bbx, to_size=size)
        self.generated_imgs_test_cropped, self.generated_imgs_test_cropped_names = self.load_generated_augmentations_by_type(
            GENERATED_IMAGES_TYPES_TEST, self.generated_dir_path, img_bbx, to_size=size)

        self.gen_imgs_train_succeeded_attacked = np.zeros_like(self.generated_imgs_train_cropped_names)
        self.gen_imgs_test_succeeded_attacked = np.zeros_like(self.generated_imgs_test_cropped_names)

        # Setting output_dir folders
        self.output_dir = f'{experiment_dir}/{img_name}'
        self.output_dir_normal = os.path.join(self.output_dir, ATTACK_TYPE_A)
        self.output_dir_special = os.path.join(self.output_dir, ATTACK_TYPE_B)  # with diffusion
        os.makedirs(self.output_dir_normal, exist_ok=True)
        os.makedirs(self.output_dir_special, exist_ok=True)

        self.generated_images_for_normal_attack = None  # generated_imgs_cropped
        self.generated_images_for_special_attack = self.generated_imgs_train_cropped  # generated_imgs_cropped

    def diffusion_images_for_attack(self):
        for img, img_name in zip(self.generated_imgs_train_cropped, self.generated_imgs_train_cropped_names):
            yield img, img_name

    def diffusion_images_for_test(self):
        for img, img_name in zip(self.generated_imgs_test_cropped, self.generated_imgs_test_cropped_names):
            yield img, img_name
    def load_generated_augmentations_by_type(self, types: str, dir_path: str, bbx: List[int], to_size: int = 32):

        generated_imgs = []
        generated_imgs_names = []
        for img_name in sorted(os.listdir(dir_path)):
            if img_name.lower().endswith(('jpg', 'png')) and img_name[:-4].split('_')[0] in types:
                # img_file_name_without_ext = img_file[:-4]
                # image_filename = img_file_name_without_ext + '.png'
                image_path = os.path.join(dir_path, img_name)
                image = cv2.imread(image_path)
                cropped_img = crop_image(image, bbx[0], bbx[1], bbx[2], bbx[3])
                cropped_resized = cv2.resize(cropped_img, (to_size, to_size))
                generated_imgs_names.append(img_name[:-4])
                generated_imgs.append(cropped_resized)

        return generated_imgs, generated_imgs_names

    def all_train_images_were_successfully_attacked(self):
        """
        Check if all generated train images by diffusion model were successfully attacked
        Returns:
            True if all of them were successfully attacked
        """
        answer = self.gen_imgs_train_succeeded_attacked.sum() == len(self.gen_imgs_train_succeeded_attacked)

        return answer
    def set_gen_train_success(self, ind: int, adv_img: np.ndarray):
        self.gen_imgs_train_succeeded_attacked[ind] = 1
        img_name = self.generated_imgs_train_cropped_names[ind]
        Image.fromarray(adv_img).save(fr'{self.save_dir}/{img_name}.png')

    def all_test_images_were_successfully_attacked(self):
        """
        Check if all generated test images by diffusion model were successfully attacked
        Returns:
            True if all of them were successfully attacked
        """
        answer = self.gen_imgs_test_succeeded_attacked.sum() == len(self.gen_imgs_test_succeeded_attacked)

        return answer