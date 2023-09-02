import argparse
import json
import os
from typing import List

import cv2
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
import sys
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from ShadowAttack.utils import pre_process_image, draw_shadow, shadow_edge_blur

from ShadowAttack import lisa, gtsrb
from ShadowAttack.pso import PSO
from settings import GENERATED_IMAGES_TYPES_TRAIN, GENERATED_IMAGES_TYPES_TEST, DF_RESULTS_COLUMNS
from load_images import process_image, plot_triple_images_and_adv, crop_image
from plot_images import plot_2_images_in_a_row
from shadow import Shadow

sys.path.append('ShadowAttack')
from ShadowAttack.utils import brightness, judge_mask_type, load_mask
from seed import *
# LISA stop sign label is: 12
# GTSRB stop sign label is: 14

parser = argparse.ArgumentParser(description="Adversarial attack by shadow")
parser.add_argument("--shadow_level", type=float, default=0.43,
                    help="shadow coefficient k")
parser.add_argument("--attack_db", type=str, default="LISA",
                    help="the target dataset should be specified for a digital attack")
parser.add_argument("--attack_type", type=str, default="physical",
                    help="digital attack or physical attack")
# parser.add_argument("--image_path", type=str, default="./xxx",
#                     help="a file path to the target image should be specified for a physical attack")
parser.add_argument("--mask_path", type=str, default="octagon_mask.png",
                    help="a file path to the mask should be specified for a physical attack")
# parser.add_argument("--image_label", type=int, default=0,
#                     help="a ground truth should be specified for a physical attack")
parser.add_argument("--polygon", type=int, default=3,
                    help="shadow shape: n-sided polygon")
parser.add_argument("--n_try", type=int, default=5,
                    help="n-random-start strategy: retry n times")
parser.add_argument("--target_model", type=str, default="normal",
                    help="attack normal model or robust model")

args = parser.parse_args()
shadow_level = args.shadow_level
target_model = args.target_model
attack_db = args.attack_db
attack_type = args.attack_type
# image_path = args.image_path
mask_path = args.mask_path
# image_label = args.image_label
polygon = args.polygon
n_try = args.n_try

# def load_params():
with open('ShadowAttack/params.json', 'rb') as f:
    params = json.load(f)
    class_n_gtsrb = params['GTSRB']['class_n']
    class_n_lisa = params['LISA']['class_n']
    device = params['device']
    position_list, mask_list = load_mask()

        # return position_list, mask_list

assert attack_type in ['digital', 'physical']
if attack_type == 'digital':
    particle_size = 10
    iter_num = 100
    x_min, x_max = -16, 48
    max_speed = 1.5
else:
    particle_size = 10
    iter_num = 200
    x_min, x_max = -112, 336
    max_speed = 10.
    n_try = 1

# def load_model(attack_db, class_n_lisa, class_n_gtsrb, device):
assert attack_db in ['LISA', 'GTSRB']
if attack_db == "LISA":
    model = lisa.LisaCNN(n_class=class_n_lisa).to(device)
    model.load_state_dict(
        # torch.load(f'ShadowAttack/model/{"adv_" if target_model == "robust" else ""}model_lisa.pth',
        torch.load(f'ShadowAttack/model/{"adv_" if target_model == "robust" else ""}model_lisa.pth',
                   map_location=torch.device(device)))
    pre_process = transforms.Compose([transforms.ToTensor()])
else:
    model = gtsrb.GtsrbCNN(n_class=class_n_gtsrb).to(device)
    model.load_state_dict(
        # torch.load(f'ShadowAttack/model/{"adv_" if target_model == "robust" else ""}model_gtsrb.pth',
        torch.load(f'ShadowAttack/model/{"adv_" if target_model == "robust" else ""}model_gtsrb.pth',
                   map_location=torch.device(device)))
    pre_process = transforms.Compose([
        pre_process_image, transforms.ToTensor()])
model.eval()

def attack(with_EOT, attack_image, label, coords, targeted_attack=False, physical_attack=False, generated_images=None, **parameters):
    """
    Physical-world adversarial attack by shadow.

    Args:
        generated_images: images generated by diffusion model or other methods
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        physical_attack: Physical / digital attack.

    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
    """
    num_query = 0
    succeed = False
    global_best_solution = float('inf')
    global_best_position = None

    for attempt in range(n_try):

        if succeed:
            break

        print(f"try {attempt + 1}:", end=" ")

        pso = PSO(with_EOT, polygon*2, particle_size, iter_num, x_min, x_max, max_speed,
                  shadow_level, attack_image, coords, model, targeted_attack,
                  physical_attack, label, pre_process, generated_images, **parameters)

        best_solution, best_pos, succeed, query = pso.update_digital() \
            if not physical_attack else pso.update_physical()

        if targeted_attack:
            best_solution = 1 - best_solution
        print(f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}")
        if best_solution < global_best_solution:
            global_best_solution = best_solution
            global_best_position = best_pos
        num_query += query

    adv_image, shadow_area = draw_shadow(
        global_best_position, attack_image, coords, shadow_level)
    coefficient = 3
    adv_image = shadow_edge_blur(adv_image, shadow_area, coefficient)

    return adv_image, succeed, num_query, Shadow(global_best_position, coords, shadow_level, coefficient)

def attack_digital(attack_db: str = "LISA"):
    shadow_level = 0.5  # Replace with the desired shadow level
    # position_list, mask_list = load_params()
    save_dir = f'./adv_img/{attack_db}/{int(shadow_level * 100)}'
    try:
        os.makedirs(save_dir, exist_ok=True)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    save_dir_bmp = os.path.join(save_dir, 'bmp')
    save_dir_png = os.path.join(save_dir, 'png')
    os.makedirs(save_dir_bmp)
    os.makedirs(save_dir_png)

    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx = process_image('kaggle_images', #/workspace/traffic-diffusion/
                                                            'kaggle_annotations', attack_db) #/workspace/traffic-diffusion/
    cnt_attacked = 0
    print(f"Start attacking {len(cropped_resized_imgs)} images")
    for index in range(len(cropped_resized_imgs)):
        mask_type = judge_mask_type(attack_db, labels[index])
        if brightness(cropped_resized_imgs[index], mask_list[mask_type]) >= 120:
            adv_img, success, num_query, shadow_params = attack(False,
                cropped_resized_imgs[index], labels[index], position_list[mask_type])

            cv2.imwrite(f"{save_dir_bmp}/{file_names[index]}_{labels[index]}_{num_query}_{success}.bmp", adv_img)
            cv2.imwrite(f"{save_dir_png}/{file_names[index]}_{labels[index]}_{num_query}_{success}.png", adv_img)
            cnt_attacked+=1

            # Plot one or two examples
            if cnt_attacked < 2:
                plot_triple_images_and_adv(orig_imgs[index], cropped_imgs[index], cropped_resized_imgs[index], adv_img)
        else:
            print(f"Skip index {index} because of brightness")

    print("Attack finished! Success rate: ", end='')
    print(Counter(map(lambda x: x[:-4].split('_')[-1],
                      os.listdir(save_dir)))['True'] / len(os.listdir(save_dir)))

def load_generated_augmentations(dir_path, bbx, to_size=32):
    generated_imgs = []
    for img_name in os.listdir(dir_path):
        if img_name.lower().endswith(('jpg','png')):
            # img_file_name_without_ext = img_file[:-4]
            # image_filename = img_file_name_without_ext + '.png'
            image_path = os.path.join(dir_path, img_name)
            image = cv2.imread(image_path)
            cropped_img = crop_image(image, bbx[0], bbx[1], bbx[2], bbx[3])
            cropped_resized = cv2.resize(cropped_img, (to_size, to_size))
            generated_imgs.append(cropped_resized)

    return generated_imgs

def load_generated_augmentations_by_type(types: str, dir_path: str, bbx: List[int], to_size: int =32):
    generated_imgs = []
    generated_imgs_names = []
    for img_name in sorted(os.listdir(dir_path)):
        if img_name.lower().endswith(('jpg','png')) and img_name[:-4].split('_')[0] in types:
            # img_file_name_without_ext = img_file[:-4]
            # image_filename = img_file_name_without_ext + '.png'
            image_path = os.path.join(dir_path, img_name)
            image = cv2.imread(image_path)
            cropped_img = crop_image(image, bbx[0], bbx[1], bbx[2], bbx[3])
            cropped_resized = cv2.resize(cropped_img, (to_size, to_size))
            generated_imgs_names.append(img_name[:-4])
            generated_imgs.append(cropped_resized)

    return generated_imgs, generated_imgs_names

# def attack_physical(attack_db):
#     global position_list
#     file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx = process_image('larger_images/image_inputs',    #kaggle_images',
#                                                                                            # /workspace/traffic-`diffusion/
#                                                                                            'larger_images/image_annotations',
#                                                                                            attack_db, crop_size=224)  # /workspace/traffic-diffusion/
#     cnt_attacked = 0
#     img_name = "road_1"
#     ind = file_names.index(img_name)
#     print(file_names[ind])
#
#     # image_path = r'kaggle_images/road66.png'
#     image = cropped_imgs[ind]
#     # image_path = r'ShadowAttack/tmp/gtsrb_30.png'
#     image_label = 12#14#1#12
#     # mask_path = r'ShadowAttack/tmp/gtsrb_30_mask.png'
#     mask_path = r'octagon_mask.png'
#     size=224
#     mask_image = cv2.resize(
#         cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (size, size))
#     mask_image = np.expand_dims(mask_image, axis=-1)
#     target_image = cv2.resize(image, (size, size))
#     # target_image = cv2.resize(
#     #     cv2.imread(image_path), (224, 224))
#     generated_dir_path = f'larger_images/image_outputs/{img_name}'   #generated_images/road53'
#     generated_imgs_cropped = load_generated_augmentations(generated_dir_path, bbx[ind], to_size=size)
#     output_dir='larger_images/physical_attack_tmp'
#     os.makedirs(output_dir, exist_ok=True)
#     pos_list = np.where(mask_image.sum(axis=2) > 0)
#     generated_images = None #generated_imgs_cropped
#     transform_num = 0#43
#     # EOT is included in the first stage
#     adv_img, _, _, shadow_params = attack(target_image, image_label, pos_list,
#                                           physical_attack=True, transform_num=transform_num, generated_images=generated_images)#generated_imgs_cropped
#
#     cv2.imwrite(f'./{output_dir}/temp.bmp', adv_img)
#     if attack_db == 'LISA':
#         predict, failed, msg = lisa.test_single_image(
#             f'./{output_dir}/temp.bmp', image_label, target_model == "robust")
#     else:
#         predict, failed = gtsrb.test_single_image(
#             f'./{output_dir}/temp.bmp', image_label, target_model == "robust")
#     if failed:
#         print('Attack failed! Try to run again.')
#
#     print("part b-------------------------------------------------")
#     # Predict stabilization
#     adv_img, _, _, shadow_params = attack(target_image, image_label, pos_list, targeted_attack=True,
#                                           physical_attack=True, target=predict, transform_num=transform_num, generated_images=generated_images) #generated_imgs_cropped
#
#     plt.imshow(cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB))
#     plt.show()
#     #---------------------------------------------------
#     # Apply shadow to test image
#     test_image = generated_imgs_cropped[0]
#     test_image_shadow, shadow_area = draw_shadow(
#         shadow_params.global_best_position, test_image, shadow_params.coords, shadow_params.shadow_level)
#     coefficient = 3
#     test_image_shadow = shadow_edge_blur(test_image_shadow, shadow_area, coefficient)
#     plt.imshow(cv2.cvtColor(test_image_shadow, cv2.COLOR_BGR2RGB))
#     plt.show()
#     #---------------------------------------------------
#
#
#     cv2.imwrite(f'./{output_dir}/adv_img.png', adv_img)
#     if attack_db == 'LISA':
#         predict, failed, msg = lisa.test_single_image(
#             f'./{output_dir}/adv_img.png', image_label, target_model == "robust")
#     else:
#         predict, failed = gtsrb.test_single_image(
#             f'./{output_dir}/adv_img.png', image_label, target_model == "robust")
#     if failed:
#         summary_msg = 'Attack failed! Try to run again.'
#     else:
#         summary_msg = 'Attack succeed! Try to implement it in the real world.'
#         print(summary_msg)
#
#
#     # cv2.waitKey(0)
#     with open(f'./{output_dir}/results.txt', 'w') as f:
#         f.write(msg)
#         f.write('\n')
#         f.write(summary_msg)
#     print("output dir: ", output_dir)

def predict_image(image, description="", print_results=True, attack_db='LISA'):
    with torch.no_grad():
        img_ = cv2.resize(image, (32, 32))
        if attack_db == 'GTSRB':
            img_ = pre_process_image(img_).astype(np.float32)
        img_ = transforms.ToTensor()(img_)
        img_ = img_.unsqueeze(0).to(device)
        predict_ = torch.softmax(model(img_), 1)
        if print_results:
            print('-'*15)
            print(description)
            print(predict_.max(1)[0].item())
            print(predict_.max(1)[1].item())

        return predict_.max(1)

def calculate_average_prob(lst, true_label, desired_wrong_label=None):
    total_desired_wrong_predictions = 0
    total_prob_sum = 0 # of desired wrong pred
    total_true_label = 0

    for type_name, pred_label, pred_prob in lst:
        if pred_label == true_label:  # meaning attack failed
            total_true_label += 1
        # elif desired_wrong_label is not None and pred_label == desired_wrong_label:
        #     total_desired_wrong_predictions += 1
        #     total_prob_sum += pred_prob
        else:
            total_desired_wrong_predictions += 1
            total_prob_sum += pred_prob




    if total_desired_wrong_predictions > 0:
        average_prob = total_prob_sum / total_desired_wrong_predictions
    else:
        average_prob = 0

    return total_desired_wrong_predictions, average_prob, total_true_label

def attack_physical(attack_db):
    global position_list
    P_PROB = 0
    P_LABEL = 1
    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx = process_image('larger_images/image_inputs',    #kaggle_images',
                                                                                           # /workspace/traffic-`diffusion/
                                                                                           'larger_images/image_annotations',
                                                                                           attack_db, crop_size=224)  # /workspace/traffic-diffusion/
    parent_dir = f'larger_images/physical_attack_3_train_3_test'
    image_label = 12 if attack_db=="LISA" else 14
    cnt_attacked = 0
    df_results = pd.DataFrame(columns=DF_RESULTS_COLUMNS)
    # mask_path = r'ShadowAttack/tmp/gtsrb_30_mask.png'
    mask_path = r'octagon_mask.png'
    size=224
    transform_num_for_normal_attack = 0#14#43
    transform_num_for_special_attack =0#2#43
    with_EOT = True
    mask_image = cv2.resize(
        cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (size, size))
    mask_image = np.expand_dims(mask_image, axis=-1)
    pos_list = np.where(mask_image.sum(axis=2) > 0)
    # img_name = "road_1"
    # ind = file_names.index(img_name)
    for ind in range(len(file_names)):
        true_label = labels[ind]
        img_name = file_names[ind]
        print(file_names[ind])
        # image_path = r'kaggle_images/road66.png'
        image = cropped_imgs[ind]
        # image_path = r'ShadowAttack/tmp/gtsrb_30.png'
        target_image = cv2.resize(image, (size, size))
        # target_image = cv2.resize(
        #     cv2.imread(image_path), (224, 224))

        # Loading generated images
        generated_dir_path = f'larger_images/image_outputs/{img_name}'   #generated_images/road53'
        generated_imgs_train_cropped, generated_imgs_train_cropped_names = load_generated_augmentations_by_type(GENERATED_IMAGES_TYPES_TRAIN, generated_dir_path, bbx[ind], to_size=size)
        generated_imgs_test_cropped, generated_imgs_test_cropped_names = load_generated_augmentations_by_type(GENERATED_IMAGES_TYPES_TEST, generated_dir_path, bbx[ind], to_size=size)

        # Setting output_dir folders
        output_dir=f'{parent_dir}/{img_name}'
        output_dir_normal = os.path.join(output_dir, 'normal_atatck')
        output_dir_special = os.path.join(output_dir, 'special_atatck') #with diffusion
        os.makedirs(output_dir_normal, exist_ok=True)
        os.makedirs(output_dir_special, exist_ok=True)

        generated_images_for_normal_attack = None #generated_imgs_cropped
        generated_images_for_special_attack = generated_imgs_train_cropped #generated_imgs_cropped
        # EOT is included in the first stage
        adv_img_normal, _, _, shadow_params_normal = attack(with_EOT, target_image, image_label, pos_list,
                                                            physical_attack=True, transform_num=transform_num_for_normal_attack,
                                                            generated_images=generated_images_for_normal_attack)#generated_imgs_cropped

        adv_img_special, _, _, shadow_params_special = attack(with_EOT, target_image, image_label, pos_list,
                                                              physical_attack=True, transform_num=transform_num_for_special_attack,
                                                              generated_images=generated_images_for_special_attack)

        predict_normal = save_temporarily_attack(adv_img_normal, attack_db, image_label, output_dir_normal)
        predict_special = save_temporarily_attack(adv_img_special, attack_db, image_label, output_dir_special)
        plot_2_images_in_a_row(cv2.cvtColor(adv_img_normal, cv2.COLOR_BGR2RGB), cv2.cvtColor(adv_img_special, cv2.COLOR_BGR2RGB), "adv_img_normal", "adv_img_special", save_path=os.path.join(output_dir, 'adv_orig_cmp.png'), plot=False)

        print("part b-------------------------------------------------")
        # Predict stabilization
        # adv_img_normal, _, _, shadow_params_normal = attack(with_EOT, target_image, image_label, pos_list, targeted_attack=True,
        #                                                     physical_attack=True, target=predict_normal, transform_num=transform_num_for_normal_attack,
        #                                                     generated_images=generated_images_for_normal_attack) #generated_imgs_cropped
        #
        # adv_img_special, _, _, shadow_params_special = attack(with_EOT, target_image, image_label, pos_list, targeted_attack=True,
        #                                                       physical_attack=True, target=predict_special, transform_num=transform_num_for_special_attack,
        #                                                       generated_images=generated_images_for_special_attack)
        # plot_2_images_in_a_row(cv2.cvtColor(adv_img_normal, cv2.COLOR_BGR2RGB), cv2.cvtColor(adv_img_special, cv2.COLOR_BGR2RGB), "adv_img_normal", "adv_img_special", save_path=os.path.join(output_dir, 'adv_orig_cmp.png'), plot=False)
        #
        #---------------------------------------------------
        with torch.no_grad():
            clean_image_predict = predict_image(target_image, "clean image", print_results=False, attack_db=attack_db)
            adv_img_normal_predict = predict_image(adv_img_normal, "adv image normal", print_results=False, attack_db=attack_db)
            adv_img_special_predict = predict_image(adv_img_special, "adv image special", print_results=False, attack_db=attack_db)
            df_result_row = [file_names[ind], labels[ind], clean_image_predict[1].item(), clean_image_predict[0].item(),
                             adv_img_normal_predict[1].item(), adv_img_normal_predict[0].item(),
                             adv_img_special_predict[1].item(), adv_img_special_predict[0].item()]

            test_gen_normal_preds, test_gen_special_preds = [], []
        # Apply shadow to test image

        for gen_type in GENERATED_IMAGES_TYPES_TEST:
            gen_ind = GENERATED_IMAGES_TYPES_TEST.index(gen_type)
            gen_test_image = generated_imgs_test_cropped[gen_ind]
            test_image_clean_predict = predict_image(gen_test_image, "test_image_clean_predict", attack_db=attack_db)

            test_image_shadow_normal = add_shadow_attack_to_image(shadow_params_normal, gen_test_image)
            test_image_shadow_normal_predict = predict_image(test_image_shadow_normal, "test_image_shadow_normal", attack_db=attack_db)
            test_gen_normal_preds.append([generated_imgs_test_cropped_names[gen_ind], test_image_shadow_normal_predict[1].item(), test_image_shadow_normal_predict[0].item()])
            cv2.imwrite(f'./{output_dir_normal}/{generated_imgs_test_cropped_names[gen_ind]}_adv_normal.png', test_image_shadow_normal)

            test_image_shadow_special = add_shadow_attack_to_image(shadow_params_special, gen_test_image)
            test_image_shadow_special_predict = predict_image(test_image_shadow_special, "test_image_shadow_special", print_results=False, attack_db=attack_db)
            test_gen_special_preds.append([generated_imgs_test_cropped_names[gen_ind], test_image_shadow_special_predict[1].item(), test_image_shadow_special_predict[0].item()])
            cv2.imwrite(f'./{output_dir_special}/{generated_imgs_test_cropped_names[gen_ind]}_adv_special.png', test_image_shadow_special)

            df_result_row += [test_image_clean_predict[1].item(), test_image_clean_predict[0].item(),
                              test_image_shadow_normal_predict[1].item(), test_image_shadow_normal_predict[0].item(),
                              test_image_shadow_special_predict[1].item(), test_image_shadow_special_predict[0].item()]
            plot_2_images_in_a_row(cv2.cvtColor(test_image_shadow_normal, cv2.COLOR_BGR2RGB), cv2.cvtColor(test_image_shadow_special, cv2.COLOR_BGR2RGB), "test_image_shadow_normal", "test_image_shadow_special", save_path=os.path.join(output_dir, f'{generated_imgs_test_cropped_names[gen_ind]}_adv_cmp.png'), plot=False)

        # Calculate total summary for each attack row on generated images:
        with open(os.path.join(output_dir_normal, 'gen_results.json'), 'w') as json_file:
            json.dump(test_gen_normal_preds, json_file)
        with open(os.path.join(output_dir_special, 'gen_results.json'), 'w') as json_file:
            json.dump(test_gen_special_preds, json_file)

        total_desired_wrong_predictions_normal, average_prob_normal, total_true_label_normal = calculate_average_prob(test_gen_normal_preds, true_label, adv_img_normal_predict[1].item())
        total_desired_wrong_predictions_special, average_prob_special, total_true_label_special = calculate_average_prob(test_gen_special_preds, true_label, adv_img_special_predict[1].item())
        df_result_row += [len(GENERATED_IMAGES_TYPES_TEST)]
        df_result_row += [total_desired_wrong_predictions_normal, average_prob_normal, total_true_label_normal]
        df_result_row += [total_desired_wrong_predictions_special, average_prob_special, total_true_label_special]

        pd.DataFrame([df_result_row], columns=DF_RESULTS_COLUMNS).to_csv(os.path.join(output_dir, f'{file_names[ind]}.csv'), index=False)
        df_results.loc[ind] = df_result_row

        #---------------------------------------------------
        df_results.to_csv(os.path.join(parent_dir, 'results.csv'), index=False)
        msg_normal, summary_msg_normal = check_and_save_final_attacked_image(adv_img_normal, attack_db, image_label, output_dir_normal)
        msg_special, summary_msg_special = check_and_save_final_attacked_image(adv_img_special, attack_db, image_label, output_dir_special)

        # cv2.waitKey(0)
        with open(f'./{output_dir}/results.txt', 'w') as f:
            f.write("normal atatck:")
            f.write(msg_normal)
            f.write('\n')
            f.write(summary_msg_normal)
            f.write('\n')
            f.write("special atatck:")
            f.write(msg_special)
            f.write('\n')
            f.write(summary_msg_special)
        print("output dir: ", output_dir)

        # if ind==2:
        #     break

def attack_physical_untargeted_only(attack_db):
    global position_list
    print("attack_db: ", attack_db)
    P_PROB = 0
    P_LABEL = 1
    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped = process_image('larger_images/image_inputs',    #kaggle_images',
                                                                                           # /workspace/traffic-`diffusion/
                                                                                           'larger_images/image_annotations',
                                                                                           attack_db, crop_size=224, mask_folder=r'larger_images/image_masks')  # /workspace/traffic-diffusion/
    with_EOT = False
    parent_dir = f'larger_images/physical_attack_untar_mask_equal_split_{attack_db}_EOT-{with_EOT}_iter-{iter_num}'
    image_label = 12 if attack_db=="LISA" else 14
    cnt_attacked = 0
    df_results = pd.DataFrame(columns=DF_RESULTS_COLUMNS)
    # mask_path = r'ShadowAttack/tmp/gtsrb_30_mask.png'
    mask_path = r'octagon_mask.png'
    size=224
    transform_num_for_normal_attack = 0#14#43
    transform_num_for_special_attack =0#2#43
    # mask_image = cv2.resize(
    #     cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (size, size))
    # img_name = "road_1"
    # ind = file_names.index(img_name)
    for ind in range(len(file_names)):
        mask_image = np.expand_dims(masks_cropped[ind], axis=-1)
        pos_list = np.where(mask_image.sum(axis=2) > 0)
        true_label = labels[ind]
        img_name = file_names[ind]
        print(file_names[ind])
        # image_path = r'kaggle_images/road66.png'
        image = cropped_imgs[ind]
        # image_path = r'ShadowAttack/tmp/gtsrb_30.png'
        target_image = cv2.resize(image, (size, size))
        # target_image = cv2.resize(
        #     cv2.imread(image_path), (224, 224))

        # Loading generated images
        generated_dir_path = f'larger_images/image_outputs/{img_name}'   #generated_images/road53'
        generated_imgs_train_cropped, generated_imgs_train_cropped_names = load_generated_augmentations_by_type(GENERATED_IMAGES_TYPES_TRAIN, generated_dir_path, bbx[ind], to_size=size)
        generated_imgs_test_cropped, generated_imgs_test_cropped_names = load_generated_augmentations_by_type(GENERATED_IMAGES_TYPES_TEST, generated_dir_path, bbx[ind], to_size=size)

        # Setting output_dir folders
        output_dir=f'{parent_dir}/{img_name}'
        output_dir_normal = os.path.join(output_dir, 'normal_atatck')
        output_dir_special = os.path.join(output_dir, 'special_atatck') #with diffusion
        os.makedirs(output_dir_normal, exist_ok=True)
        os.makedirs(output_dir_special, exist_ok=True)

        generated_images_for_normal_attack = None #generated_imgs_cropped
        generated_images_for_special_attack = generated_imgs_train_cropped #generated_imgs_cropped
        # EOT is included in the first stage
        adv_img_normal, _, _, shadow_params_normal = attack(with_EOT, target_image, image_label, pos_list,
                                                            physical_attack=True, transform_num=transform_num_for_normal_attack,
                                                            generated_images=generated_images_for_normal_attack)#generated_imgs_cropped

        adv_img_special, _, _, shadow_params_special = attack(with_EOT, target_image, image_label, pos_list,
                                                              physical_attack=True, transform_num=transform_num_for_special_attack,
                                                              generated_images=generated_images_for_special_attack)

        # predict_normal = save_temporarily_attack(adv_img_normal, attack_db, image_label, output_dir_normal)
        # predict_special = save_temporarily_attack(adv_img_special, attack_db, image_label, output_dir_special)
        plot_2_images_in_a_row(cv2.cvtColor(adv_img_normal, cv2.COLOR_BGR2RGB), cv2.cvtColor(adv_img_special, cv2.COLOR_BGR2RGB), "adv_img_normal", "adv_img_special", save_path=os.path.join(output_dir, 'adv_orig_cmp.png'), plot=False)

        print("part b-------------------------------------------------")
        # # Predict stabilization
        # adv_img_normal, _, _, shadow_params_normal = attack(with_EOT, target_image, image_label, pos_list, targeted_attack=True,
        #                                                     physical_attack=True, target=predict_normal, transform_num=transform_num_for_normal_attack,
        #                                                     generated_images=generated_images_for_normal_attack) #generated_imgs_cropped
        #
        # adv_img_special, _, _, shadow_params_special = attack(with_EOT, target_image, image_label, pos_list, targeted_attack=True,
        #                                                       physical_attack=True, target=predict_special, transform_num=transform_num_for_special_attack,
        #                                                       generated_images=generated_images_for_special_attack)
        # plot_2_images_in_a_row(cv2.cvtColor(adv_img_normal, cv2.COLOR_BGR2RGB), cv2.cvtColor(adv_img_special, cv2.COLOR_BGR2RGB), "adv_img_normal", "adv_img_special", save_path=os.path.join(output_dir, 'adv_orig_cmp.png'), plot=False)
        #---------------------------------------------------
        with torch.no_grad():
            clean_image_predict = predict_image(target_image, "clean image", print_results=False, attack_db=attack_db)
            adv_img_normal_predict = predict_image(adv_img_normal, "adv image normal", print_results=False, attack_db=attack_db)
            adv_img_special_predict = predict_image(adv_img_special, "adv image special", print_results=False, attack_db=attack_db)
            df_result_row = [file_names[ind], labels[ind], clean_image_predict[1].item(), clean_image_predict[0].item(),
                             adv_img_normal_predict[1].item(), adv_img_normal_predict[0].item(),
                             adv_img_special_predict[1].item(), adv_img_special_predict[0].item()]

            test_gen_normal_preds, test_gen_special_preds = [], []
        # Apply shadow to test image


        for gen_type in GENERATED_IMAGES_TYPES_TEST:
            # for i in range(1,3):
            #     gen_ind=0
            #
            gen_ind = generated_imgs_test_cropped_names.index(gen_type)
            for i in range(1,3):
                # gen_ind = GENERATED_IMAGES_TYPES_TEST.index(gen_type)
                gen_test_image = generated_imgs_test_cropped[gen_ind]
                test_image_clean_predict = predict_image(gen_test_image, "test_image_clean_predict", attack_db=attack_db)

                test_image_shadow_normal = add_shadow_attack_to_image(shadow_params_normal, gen_test_image)
                test_image_shadow_normal_predict = predict_image(test_image_shadow_normal, "test_image_shadow_normal", attack_db=attack_db)
                test_gen_normal_preds.append([generated_imgs_test_cropped_names[gen_ind], test_image_shadow_normal_predict[1].item(), test_image_shadow_normal_predict[0].item()])
                cv2.imwrite(f'./{output_dir_normal}/{generated_imgs_test_cropped_names[gen_ind]}_adv_normal.png', test_image_shadow_normal)

                test_image_shadow_special = add_shadow_attack_to_image(shadow_params_special, gen_test_image)
                test_image_shadow_special_predict = predict_image(test_image_shadow_special, "test_image_shadow_special", print_results=False, attack_db=attack_db)
                test_gen_special_preds.append([generated_imgs_test_cropped_names[gen_ind], test_image_shadow_special_predict[1].item(), test_image_shadow_special_predict[0].item()])
                cv2.imwrite(f'./{output_dir_special}/{generated_imgs_test_cropped_names[gen_ind]}_adv_special.png', test_image_shadow_special)

                df_result_row += [test_image_clean_predict[1].item(), test_image_clean_predict[0].item(),
                                  test_image_shadow_normal_predict[1].item(), test_image_shadow_normal_predict[0].item(),
                                  test_image_shadow_special_predict[1].item(), test_image_shadow_special_predict[0].item()]
                plot_2_images_in_a_row(cv2.cvtColor(test_image_shadow_normal, cv2.COLOR_BGR2RGB), cv2.cvtColor(test_image_shadow_special, cv2.COLOR_BGR2RGB), "test_image_shadow_normal", "test_image_shadow_special", save_path=os.path.join(output_dir, f'{generated_imgs_test_cropped_names[gen_ind]}_adv_cmp.png'), plot=False)
                gen_ind+=1

        # Calculate total summary for each attack row on generated images:
        with open(os.path.join(output_dir_normal, 'gen_results.json'), 'w') as json_file:
            json.dump(test_gen_normal_preds, json_file)
        with open(os.path.join(output_dir_special, 'gen_results.json'), 'w') as json_file:
            json.dump(test_gen_special_preds, json_file)

        total_desired_wrong_predictions_normal, average_prob_normal, total_true_label_normal = calculate_average_prob(test_gen_normal_preds, true_label, adv_img_normal_predict[1].item())
        total_desired_wrong_predictions_special, average_prob_special, total_true_label_special = calculate_average_prob(test_gen_special_preds, true_label, adv_img_special_predict[1].item())
        df_result_row += [len(GENERATED_IMAGES_TYPES_TEST)*2]
        df_result_row += [total_desired_wrong_predictions_normal, average_prob_normal, total_true_label_normal]
        df_result_row += [total_desired_wrong_predictions_special, average_prob_special, total_true_label_special]

        pd.DataFrame([df_result_row], columns=DF_RESULTS_COLUMNS).to_csv(os.path.join(output_dir, f'{file_names[ind]}.csv'), index=False)
        df_results.loc[ind] = df_result_row

        #---------------------------------------------------
        df_results.to_csv(os.path.join(parent_dir, 'results.csv'), index=False)
        msg_normal, summary_msg_normal = check_and_save_final_attacked_image(adv_img_normal, attack_db, image_label, output_dir_normal)
        msg_special, summary_msg_special = check_and_save_final_attacked_image(adv_img_special, attack_db, image_label, output_dir_special)

        # cv2.waitKey(0)
        with open(f'./{output_dir}/results.txt', 'w') as f:
            f.write("normal atatck:")
            f.write(msg_normal)
            f.write('\n')
            f.write(summary_msg_normal)
            f.write('\n')
            f.write("special atatck:")
            f.write(msg_special)
            f.write('\n')
            f.write(summary_msg_special)
        print("output dir: ", output_dir)

        # if ind==2:
        #     break

def check_and_save_final_attacked_image(adv_img, attack_db, image_label, output_dir):
    cv2.imwrite(f'./{output_dir}/adv_img.png', adv_img)
    if attack_db == 'LISA':
        predict, failed, msg = lisa.test_single_image(
            f'./{output_dir}/adv_img.png', image_label, target_model == "robust")
    else:
        predict, failed, msg = gtsrb.test_single_image(
            f'./{output_dir}/adv_img.png', image_label, target_model == "robust")
    if failed:
        summary_msg = 'Attack failed! Try to run again.'
    else:
        summary_msg = 'Attack succeed! Try to implement it in the real world.'
        print(summary_msg)
    return msg, summary_msg

def add_shadow_attack_to_image(shadow_params_normal, test_image):
    test_image_shadow, shadow_area = draw_shadow(
        shadow_params_normal.global_best_position, test_image, shadow_params_normal.coords,
        shadow_params_normal.shadow_level)
    coefficient = 3
    test_image_shadow = shadow_edge_blur(test_image_shadow, shadow_area, coefficient)
    return test_image_shadow


def save_temporarily_attack(adv_img, attack_db, image_label, output_dir):
    cv2.imwrite(f'./{output_dir}/temp.bmp', adv_img)
    if attack_db == 'LISA':
        predict, failed, msg = lisa.test_single_image(
            f'./{output_dir}/temp.bmp', image_label, target_model == "robust")
    else:
        predict, failed = gtsrb.test_single_image(
            f'./{output_dir}/temp.bmp', image_label, target_model == "robust")
    if failed:
        print('Attack failed! Try to run again.')
    return predict

if __name__ == "__main__":
    # Call the attack_digital() function
    # attack_db = "LISA"  # # Replace with "LISA" or "GTSRB" depending on your use case, Replace with the actual attack database
    # attack_digital(attack_db)
    # attack_physical(attack_db)
    attack_physical_untargeted_only(attack_db)