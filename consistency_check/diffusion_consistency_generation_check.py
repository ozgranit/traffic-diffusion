import argparse
import copy
import os.path
import pickle
import cv2
import numpy as np
import random
from attacks.RFLA.our_pos_reflect_attack_thesis import loading_config_file, PSOAttack
from decorators.timeit_decorator import timeit
from diffusionImageGenerator.locally.stable_diffusion_xl_params import stable_diffusion_xl_main_params
from load_images import process_image
from prompts import prompt_getter, NEGATIVE_PROMPT
from settings import GENERATED_IMAGES_TYPES_TRAIN
from thesis.create_images_locally import load_stable_diffusion_xl, get_generator, reinitialize_generator
from buildData.input_data import InputData

import torch
from seed import *

@timeit
def run_stable_diffusion(stable_diffusion_model, generator, stable_diffusion_main_params, adv_image_large_pil, cur_prompt):
    result_image = stable_diffusion_model(
        prompt=cur_prompt + ". do not change the rectangle shadow inside the stop sign",
        # prompt=cur_prompt + ". do not change the rectangle shadow in the middle of the stop sign",
        generator=generator,
        image=adv_image_large_pil, negative_prompt=NEGATIVE_PROMPT, **stable_diffusion_main_params)
    model_type = 'local'
    result_image = result_image.images[0]
    return model_type, result_image

def apply_attack_on_img(cropped_src_clean_img_resized, attack_params, src_clean_img, cropped_src_clean_img, bbx):
    xmin, ymin, xmax, ymax = bbx
    cv2.fillPoly(cropped_src_clean_img_resized, attack_params.points,
                 (attack_params.red, attack_params.green, attack_params.blue))
    adv_img = cv2.addWeighted(cropped_src_clean_img_resized, attack_params.alpha, cropped_src_clean_img_resized,
                                1 - attack_params.alpha, attack_params.gamma)
    adv_img_resized = cv2.resize(adv_img, cropped_src_clean_img.shape[:2][::-1])
    src_clean_img[ymin:ymax, xmin:xmax] = adv_img_resized
    adv_clean_large_img = src_clean_img

    return adv_img, adv_img_resized, adv_clean_large_img

def diffusion_consistency_generation_check():
    print("Starting diffusion_consistency_generation_check...")
    stable_diffusion_model = load_stable_diffusion_xl()
    generator_seed = 525901257
    generator = get_generator(generator_seed)
    # stable_diffusion_model = None
    model_params = stable_diffusion_xl_main_params()

    # Loading imgs
    parser = argparse.ArgumentParser(description="Random Search Parameters")
    ################# the file path of config.yml #################
    parser.add_argument("--yaml_file", type=str, default="attacks/RFLA/config.yml", help="the settings config")
    ################# load config.yml file    ##################
    args = loading_config_file(parser)

    input_data = InputData(args.dataset_name)
    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped = process_image(
        input_data.input_folder,
        input_data.annotations_folder,
        args.model_name, crop_size=args.image_size, mask_folder=input_data.mask_folder)

    # Load perturbation:
    attack_params_path = r'experiments/RFLA_tmp/larger_images/physical_attack_RFLA_LISA_isAdv-0_shape-hexagon_maxIter-200_ensemble-0_interploate-0/road_1/special_attack/dawn.pkl'
    with open(attack_params_path, 'rb') as reader:
        attack_params = pickle.load(reader)

    # tmp
    with open(r'experiments/RFLA_tmp/larger_images/physical_attack_RFLA_LISA_isAdv-0_shape-hexagon_maxIter-200_ensemble-0_interploate-0/road_1/special_attack/dawn_2.pkl', 'rb') as reader_tmp:
        attack_params_tmp = pickle.load(reader_tmp)

    # print(f"comparing 2 perturbation (It should be the same): {attack_params == attack_params_tmp}")

    # Set prompt type
    prompt_types = GENERATED_IMAGES_TYPES_TRAIN
    dest_path = r'experiments/check_genertion_consistency/check_2'
    os.makedirs(dest_path, exist_ok=True)
    ext = 'png'

    for ii in range(5):
        print("ii")
        curr_dest_dir = os.path.join(dest_path, str(ii))
        os.makedirs(curr_dest_dir, exist_ok=True)
        for i, (orig_img, cropped_img, cropped_resized_img, bbx_curr, mask, filename) in enumerate(
            zip(orig_imgs, cropped_imgs, cropped_resized_imgs, bbx, masks_cropped, file_names)):

            adv_images, adv_img_resized, adv_clean_large_img = apply_attack_on_img(cropped_resized_img, attack_params,
                                                                                copy.deepcopy(orig_img), cropped_img, bbx_curr)
            adv_images = np.expand_dims(adv_images, 0)

            for prompt_desc, cur_prompt in prompt_getter.items():
                if prompt_desc in prompt_types:
                    adv_images_large = PSOAttack.insert_cropped_resized_with_attack_into_large_image(adv_images, copy.deepcopy(orig_img),
                                                                                                cropped_img, bbx_curr)
                    adv_images_large_pil = PSOAttack.numpy_to_pil_img_for_diffusion_model(
                        adv_images_large)
                    adv_image_large_pil = adv_images_large_pil[0]
                    if stable_diffusion_model is not None:
                        # result_image = self.stable_diffusion_model(prompt=cur_prompt + ". do not change the hexagon shadow in the middle of the stop sign",
                        model_type, result_image = run_stable_diffusion(stable_diffusion_model, generator, model_params, adv_image_large_pil, cur_prompt)
                        reinitialize_generator(generator, generator_seed)
                    else:
                        model_type = 'api'
                        result_image = PSOAttack.generate_diffusion_image_using_api(prompt_desc=prompt_desc,
                                                            prompt=cur_prompt + ". do not change the hexagon shadow in the middle of the stop sign",
                                                            image=adv_image_large_pil,
                                                            negative_prompt=NEGATIVE_PROMPT,
                                                            params=model_params)

                    img_out_name = f'{model_type}_iter-{iter}_{prompt_desc}-{i}.png'
                    result_image_resized_cropped = PSOAttack.process_image_after_diffusion_generation(
                        copy.deepcopy(result_image), bbx_curr, adv_images[0].shape[1:3])
                    fname = filename + f'_orig_{prompt_desc}.{ext}'
                    cv2.imwrite(os.path.join(curr_dest_dir, fname), orig_img)

                    fname = filename + f'_orig_adv_{prompt_desc}.{ext}'
                    cv2.imwrite(os.path.join(curr_dest_dir, fname), adv_images_large[0])

                    adv_gen_img_large = np.array(result_image)
                    fname = filename + f'_gen_adv_{prompt_desc}.{ext}'
                    cv2.imwrite(os.path.join(curr_dest_dir, fname), adv_gen_img_large)

                    # break

            break


    print("Finished!")
# if __name__ == "__main__":
#     main()
