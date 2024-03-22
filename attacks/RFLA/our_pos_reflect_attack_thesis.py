import argparse
import base64
import copy
import json
import os
import pickle
from typing import List, Tuple, Dict

import pandas as pd
import torch
import PIL
import yaml
import time
import sys
import numpy as np
import cv2
from PIL import Image, ImageOps

# Add a directory to the Python import search path
# sys.path.append('/home/sharifm/students/ortallevi/trafic-diffusion')
# sys.path.append('/home/sharifm/students/ortallevi/trafic-diffusion/')
# sys.path.append('trafic-diffusion/')

# sys.path.append('attacks/')
# sys.path.append('attacks')
# sys.path.append('attacks/RFLA')
# print(os.curdir)
# print(os.listdir(os.curdir))
# sys.path.append('.')
# sys.path.append('/home/sharifm/students/ortallevi/trafic-diffusion/attacks')
# os.chdir('/home/sharifm/students/ortallevi/trafic-diffusion')
# os.chdir('trafic-diffusion')

from attacks.RFLA.attack_params import AttackParams
from attacks.RFLA.classes.iteration_info import IterationInfo
# from attack_params import AttackParams
from buildData.diffusion_generated_imgs import DiffusionImages
from buildData.input_data import InputData
from datasets.larger_images.larger_images_settings import LARGER_IMAGES
from decorators.timeit_decorator import timeit
from diffusionImageGenerator.create_images_api import TOKENS
from diffusionImageGenerator.locally.stable_diffusion_loader import load_stable_diffusion_xl
from diffusionImageGenerator.locally.stable_diffusion_xl_params import stable_diffusion_xl_main_params
from img_utils import crop_image
from models.gtsrb_model import GtsrbModel
from models.lisa_model import LisaModel
from models.base_model import BaseModel
from inferenceAndResults import inference_on_src_attacked
from load_images import process_image
from plot_images import create_pair_plots
from prompts import prompt_getter, NEGATIVE_PROMPT
from settings import ATTACK_TYPE_A, ATTACK_TYPE_B, LISA, GTSRB, STOP_SIGN_LISA_LABEL, STOP_SIGN_GTSRB_LABEL, DEVICE, \
    GENERATED_IMAGES_TYPES_TRAIN, GENERATED_IMAGES_TYPES_TEST
import attacks.RFLA.utils_rfla as utils_rfla
import random
import requests

from thesis.create_images_locally import get_generator, reinitialize_generator

# Set seed
# Set seed for PyTorch
# Check if CUDA is available and set seed for CUDA
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set this to False for fully deterministic results
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

# Set CUBLAS_WORKSPACE_CONFIG environment variable based on CUDA version
cuda_version = torch.version.cuda
if cuda_version is not None:
    if cuda_version >= "10.2":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

class PSOAttack(object):
    def __init__(self,
                 c_list: list,
                 omega_bound: list,
                 lower_bound: list,
                 higher_bound: list,
                 v_lower: list,
                 v_higher: list,
                 model_wrapper: List[BaseModel] = None,
                 mask=None,
                 # pre_process=None,
                 transform=None,
                 image_size=224,
                 dimension=8,
                 max_iter=10000,
                 size=50,
                 sub_size=50,
                 shape_type='tri',
                 save_dir='saved_images',
                 logs_dir='logs',
                 device = DEVICE,
                 generator_seed=525901257,
                 generator=None
                 ):

        # self.init_iterations = List[IterationInfo] = []
        # self.update_iterations = List[IterationInfo] = []
        self.dimension = dimension  # the dimension of the base variable
        self.max_iter = max_iter  # maximum iterative number
        self.size = size  # the size of the particle
        self.sub_size = sub_size  # the number of the geometrical shapes in a circle
        self.pop_bound = []  # bound container
        self.pop_bound.append(lower_bound) # lower bound of solution
        self.pop_bound.append(higher_bound) # upper bound of solution
        self.omega_bound = omega_bound  # interia weight
        self.v_bound = []  # velocity container
        self.v_bound.append(v_lower)  # lower bound of velocity
        self.v_bound.append(v_higher)  # upper bound of velocity
        self.c_list = c_list  # learning factor
        self.mask = mask    # binary mask. ImageNet: all one matrix
        self.image_size = image_size  # image size
        self.shape_type = shape_type  # type of geometry shape
        self.device = device
        self.generator_seed = generator_seed
        self.generator = generator
        self.models_wrapper = model_wrapper  # the model
        # self.model_name = model_name
        self.transform = transform  # transformation for input of the model
        # self.pre_process = pre_process
        self.save_dir = save_dir
        self.logs_dir = logs_dir
        self.best_pop_on_src_and_diffusoin = None
        if shape_type in ['line']:
            self.dimension += 0
        elif shape_type in ["triangle", "rectangle"]:
            self.dimension += 1
        elif shape_type in ["pentagon", "hexagon"]:
            self.dimension += 2

        self.reset_attack_params()

    def reset_attack_params(self):
        self.pops = np.zeros((self.size, self.sub_size, self.dimension))  # Store all solutions
        self.v = np.zeros((self.size, self.sub_size, self.dimension))  # store all velocity
        self.p_best = np.zeros((self.size, self.dimension))  # the personal best solution
        self.g_best = np.zeros((1, self.dimension))[0]  # the global best solution in terms of sum of a circle
        self.sg_best = np.zeros((1, self.dimension))[0]  # the global best solution
        self.p_best_n_success_imgs = np.zeros(self.p_best.shape[0])  # our added var
        self.g_best_fitness = 0  # store the best fitness in terms of sum of a circle
        self.sg_best_fitness = 0  # store the best fitness score
        self.p_best_fitness = [0] * self.size  # store the person best fitness
        self.best_pop_on_src_and_diffusoin: None  # our added var
        self.best_pop_on_src_and_diffusoin_count_imgs_success = 0  # our added var - on how many images (src_img, and diffusion images - the attack with the pos in 'self.best_pop_on_src_and_diffusoin' has succeeded
        self.best_diffusion_imgs_by_prompt_desc = None
        self.total_succeeded_in_iteration = [0] * self.size  # our added var - on how many images (src_img, and diffusion images - the attack with the pos in 'self.best_pop_on_src_and_diffusoin' has succeeded
        self.best_succeeded_pops_mean_scores = [-1] * self.size
        self.softmax_scores= [[] for i in range(self.size)]
        self.is_attack_succeeded_for_each_img = [[] for i in range(self.size)]

        self.lowest_softmax_for_attacked_class = np.zeros(self.p_best.shape[0])
        self.lowest_softmax_for_attacked_class_succeeded = np.zeros(self.p_best.shape[0])
        self.lowest_softmax_for_attacked_class_index = np.zeros(self.p_best.shape[0])
        self.is_lowest_softmax_for_attacked_class_in_best_chosen_pop = np.zeros(self.p_best.shape[0])

    def set_model(self, model):
        self.model = model

    def set_mask(self, mask):
        self.mask = mask

    def check_circle_in_mask(self, point_x, point_y):
        if not self.mask[point_x, point_y]:
            return False
        return True

    def get_circle_raidus(self):
        """
        Random generate a circle with point (x, y) and raidus
        :return:
        """
        radius = np.random.uniform(self.pop_bound[0][2], self.pop_bound[1][2])
        point_x = np.random.randint(radius, self.image_size - radius)
        point_y = np.random.randint(radius, self.image_size - radius)
        while not self.check_circle_in_mask(point_x, point_y):
            point_x = np.random.randint(radius, self.image_size - radius)
            point_y = np.random.randint(radius, self.image_size - radius)
        return (point_x, point_y, radius)

    def initial_per_circle(self, circle):
        """
        generate solution for a circle
        :param circle:
        :return:
        """
        x, y, r = circle
        pops = []
        # r, x, y, alpha, red, green, blue, angle, angle
        for j in range(self.sub_size):  # 50个圆中的三角形
            alpha = np.random.uniform(self.pop_bound[0][3], self.pop_bound[1][3])
            red = np.random.randint(self.pop_bound[0][4], self.pop_bound[1][4])
            green = np.random.randint(self.pop_bound[0][5], self.pop_bound[1][5])
            blue = np.random.randint(self.pop_bound[0][5], self.pop_bound[1][5])
            angle = np.random.uniform(self.pop_bound[0][6], self.pop_bound[1][6])
            if self.shape_type in 'line':
                pops.append((x, y, r, alpha, red, green, blue, angle))
            elif self.shape_type in "triangle":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta))
            elif self.shape_type in "rectangle":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta))
            elif self.shape_type in "pentagon":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                _, angle_beta1 = utils_rfla.get_new_angle((x, y), r, [angle, angle_beta])  # 
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta, angle_beta1))
            elif self.shape_type in "hexagon":
                _, angle_beta = utils_rfla.get_new_angle((x, y), r, angle)
                _, angle_beta1 = utils_rfla.get_new_angle((x, y), r, [angle, angle_beta])  # 
                pops.append((x, y, r, alpha, red, green, blue, angle, angle_beta, angle_beta1))
            else:
                raise ValueError("Please select the shape in [line, triangle, rectangle, pentagon, hexagon]")
        return pops

    def gen_adv_images_by_pops(self, image: np.ndarray, pops: List[Tuple]) -> Tuple[np.ndarray, List[AttackParams]]:
        result_images = []
        result_attacks_params = []
        for j, pop in enumerate(pops):
            image_raw = copy.deepcopy(image)
            x, y, r, alpha, red, green, blue = pop[:7]
            angles = pop[7:]
            x_0, y_0 = utils_rfla.get_point_by_angle((x, y), r, angles[0])

            if self.shape_type in 'line':
                x_1, y_1 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))
                points = np.array([(x_0, y_0), (x_1, y_1)]).astype(np.int32)
                cv2.line(image_raw, points[0], points[1], color=(red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "triangle":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r,
                                                         angle_beta)
                x_11, y_11 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))
                points = np.array([(x_0, y_0), (x_1, y_1), (x_11, y_11)]).astype(np.int32)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "rectangle":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta)
                x_11, y_11 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))
                x_21, y_21 = utils_rfla.get_symmetric_point_of_center((x, y), (x_1, y_1))
                points = np.array([(x_0, y_0), (x_1, y_1), (x_11, y_11), (x_21, y_21)]).astype(np.int32)
                points = utils_rfla.sort_points_by_distance(points)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "pentagon":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta)

                angle_beta1 = angles[2]
                x_2, y_2 = utils_rfla.get_point_by_angle((x, y), r, angle_beta1)

                x_21, y_21 = utils_rfla.get_symmetric_point_of_center((x, y), (x_2, y_2))
                x_31, y_31 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))

                points = np.array([(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_21, y_21), (x_31, y_31)]).astype(np.int32)
                points = utils_rfla.sort_points_by_distance(points)

                cv2.polylines(image_raw, [points], True, (red, green, blue), 1)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "hexagon":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta)

                angle_beta1 = angles[2]
                x_2, y_2 = utils_rfla.get_point_by_angle((x, y), r, angle_beta1)

                x_21, y_21 = utils_rfla.get_symmetric_point_of_center((x, y), (x_1, y_1))
                x_31, y_31 = utils_rfla.get_symmetric_point_of_center((x, y), (x_0, y_0))
                x_41, y_41 = utils_rfla.get_symmetric_point_of_center((x, y), (x_2, y_2))

                points = np.array(
                    [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_31, y_31), (x_21, y_21), (x_41, y_41)]).astype(np.int32)
                points = utils_rfla.sort_points_by_distance(points)

                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)
            # Image.fromarray(image_new).save("optimizing_images/image_{j}.png")
            result_images.append(image_new)
            attack_params = AttackParams([points], red, green, blue, alpha, 1-alpha, 0)
            result_attacks_params.append(attack_params)

        return np.array(result_images), result_attacks_params

    def to_tensor(self, images):
        try:
            images = np.transpose(images, (0, 3, 1, 2))
        except:
            images = np.transpose(images, (2, 0, 1))
            images = np.expand_dims(images, axis=0)
        if self.transform is not None:
            # Covert the image to PIL.Image
            images = torch.cat([self.transform(Image.fromarray(img)).unsqueeze(dim=0) for img in images])
        else:
            images = images.astype(np.float32) / 255.
            images = torch.from_numpy(images)
        return images

    def initialize(self, orig_image, cropped_image, image: np.ndarray, bbx, label: torch.Tensor, filename: str = "test",
                   diffusion_imgs: DiffusionImages = None):
        print(f"Initialize {filename}")
        image_raw = copy.deepcopy(image)
        cropped_image_raw = copy.deepcopy(cropped_image)
        orig_image_raw = copy.deepcopy(orig_image)
        temp = 1e+5
        temp_real = 1e+5
        for i in range(self.size):
            x, y, r = self.get_circle_raidus()
            pops = self.initial_per_circle((x, y, r))
            succeeded_pops = np.zeros(len(pops))
            self.pops[i, ...] = pops
            all_pops = [pops]

            circlr_v = [random.uniform(self.v_bound[0][k], self.v_bound[1][k]) for k in range(self.dimension) if k < 3]
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    if k < 3:
                        self.v[i][j][:3] = circlr_v
                        continue
                    self.v[i][j][k] = random.uniform(self.v_bound[0][k], self.v_bound[1][k])

            adv_images, adv_params = self.gen_adv_images_by_pops(image_raw, pops)
            if diffusion_imgs is not None:
                # TODO: generate images with the attack.

                adv_images, adv_large_images_by_prompt_desc = self.generate_diffusion_images_conatining_attack(adv_images, orig_image_raw,
                                                                              cropped_image_raw, bbx, filename=filename, train=True, main_itr=-1, iter=i, initialize=True)
                # adv_images, adv_params = self.gen_adv_to_multiple_imgs(adv_images, adv_params, pops, diffusion_imgs)

            softmax = self.calculate_fitness(adv_images)
            fitness_score, pred_labels = torch.max(softmax, dim=1)
            softmax_numpy = copy.deepcopy(softmax).detach().cpu().numpy()
            # Exit when the best solution is obtained
            success_indicator = (pred_labels != label)
            if diffusion_imgs is not None:
                self.add_score_and_indication_if_attack_succeeded_to_img_names(softmax_numpy[:, label],
                                                                  success_indicator, len(pops),
                                                                  train=True, filename=filename,
                                                                main_itr=-1, iter=i, initialize=True)
            # success_indicator_ = success_indicator.cpu().data.numpy()
            success_indicator_index = torch.where(success_indicator)[0]
            print(f"success_indicator_index: {success_indicator_index}")
            # self.find_best_result_for_current_iteration(softmax, success_indicator_index)
            min_index_class, smallest_score_for_class = self.get_smallest_index_and_score_for_attacking_class(softmax_numpy, label)
            smallest_score_attack_succeeded = pred_labels[min_index_class] != label

            success_indicator_index = success_indicator_index.cpu().data.numpy()
            if len(success_indicator_index) >= 1:
                print(f"pred_labels[min_index_class] != label: {pred_labels[min_index_class] != label}")
                # if success_indicator.sum().item() >= 1:
                # adv_images[(pred_labels != label).cpu().data.numpy()]
                image2saved = adv_images[success_indicator_index]
                if diffusion_imgs is not None:
                    for pop_ind in success_indicator_index:
                        # TODO: set 30 again
                        # succeeded_pops[pop_ind % 30] += 1
                        succeeded_pops[pop_ind % len(pops)] += 1

                    self.total_succeeded_in_iteration[i] = succeeded_pops.max()
                    # self.save_best_general_iteration_result(adv_images[min_index_class], adv_params[min_index_class], self.p_best[i],
                    #                                         min_index_class, smallest_score_for_class,
                    #                                         smallest_score_attack_succeeded,
                    #                                         diffusion_imgs, succeeded_pops.max())
                    if succeeded_pops.max() > self.best_pop_on_src_and_diffusoin_count_imgs_success:
                        best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.update_iteration_params(fitness_score, i, pops, succeeded_pops,
                                                     success_indicator_index, softmax_numpy[:, label])
                        self.update_best_pop_attack(i, succeeded_pops, best_succeeded_pops_ind, best_succeeded_pops_mean_score, adv_large_images_by_prompt_desc)
                        print(
                            f"Initialize Success on {filename} on {succeeded_pops.max()} imgs, i-{i}:  {self.best_pop_on_src_and_diffusoin_count_imgs_success}, g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")

                    if succeeded_pops.max() == (len(diffusion_imgs.generated_imgs_train_cropped_names) / 2) + 1:
                        best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                     success_indicator_index, softmax_numpy[:, label])
                        self.update_best_pop_attack(i, succeeded_pops, best_succeeded_pops_ind, best_succeeded_pops_mean_score, adv_large_images_by_prompt_desc)

                        # TODO: apply attack to test generated images...
                        print(
                            f"Initialize Success All {filename}, i-{i}: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                        return True
                    #---------------
                    # if succeeded_pops.max() == (len(diffusion_imgs.generated_imgs_train_cropped_names) + 1):
                    #     # TODO: apply attack to test generated images...
                    #     return True
                    else:
                        # g_fitness, p_best_idx, g_fitness_real = self.compute_best_pop_params_with_diffusion_imgs_in_equal_state(
                        #     fitness_score, pops, success_indicator_index)
                        # self.pops[i, ...] = pops
                        #Ortal

                        best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.update_iteration_params(fitness_score, i, pops, succeeded_pops, success_indicator_index, softmax_numpy[:, label])

                        # if self.p_best_n_success_imgs[i] > self.best_pop_on_src_and_diffusoin_count_imgs_success:
                        #     # setting number of attacked successfully images (of src + diffusion imgs)
                        #     self.best_pop_on_src_and_diffusoin_count_imgs_success = self.p_best_n_success_imgs[i]
                        #
                        #     self.best_pop_on_src_and_diffusoin = self.p_best[i]
                        #     self.g_best = self.p_best[i]
                        #     self.sg_best = self.p_best[i]
                        # # else:
                        # #     g_fitness_sum_score_failed, g_fitness_min_avg_score_failed, \
                        # #         g_fitness_min_avg_score_failed_ind = self.get_diffusion_score_min_and_sum_of_failed_attack(
                        # #         fitness_score, success_indicator_index)
                        # #     self.p_best[i] = pops[g_fitness_min_avg_score_failed_ind.item()]
                        # #     self.p_best_fitness[i] = g_fitness_min_avg_score_failed
                        #
                        # # return False

                    # src_img_success = True if np.floor(success_indicator_index / len(pops)) == 0 else False
                    # # Getting only index for generated images, i.e 0...5 (we filter out all original 0 because they belong to src image)
                    # diffusion_img_success_index = set([int(np.floor(ind / len(pops)) - 1) for ind in success_indicator_index if np.floor(ind / len(pops)) > 0])
                    #
                    # for ind in diffusion_img_success_index:
                    #     image2saved_ = cv2.cvtColor(image2saved[ind], cv2.COLOR_BGR2RGB)
                    #     diffusion_imgs.set_gen_train_success(ind, image2saved_)
                    #
                    #     diffusion_img_name = diffusion_imgs.generated_imgs_train_cropped_names[ind]
                    #
                    # if diffusion_imgs.all_test_images_were_successfully_attacked() and src_img_success:
                    #     return True
                else:
                    # self.update_iteration_params(fitness_score, i, pops, succeeded_pops,
                    #                              success_indicator_index)
                    self.total_succeeded_in_iteration[i] = 1
                    best_pop = self.pops[i][success_indicator_index[0]]
                    self.best_pop_on_src_and_diffusoin = best_pop
                    self.p_best[i] = best_pop
                    self.g_best = self.p_best[i]
                    self.sg_best = self.p_best[i]

                    # self.save_best_general_iteration_result(adv_images[min_index_class], adv_params[min_index_class], self.p_best[i],
                    #                                         min_index_class, smallest_score_for_class,
                    #                                         smallest_score_attack_succeeded,
                    #                                         diffusion_imgs, 1)

                    # image2saved_ = cv2.cvtColor(image2saved[0], cv2.COLOR_BGR2RGB) # TODO: check is really needed
                    # Image.fromarray(image2saved_).save(fr'{self.save_dir}/{filename}.png')
                    print(
                        f"Initialize Success {filename}: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                    return True
            else:

                if diffusion_imgs is not None:
                    best_succeeded_pops_ind, best_succeeded_pops_mean_score, mask, total_imgs_in_attack = self.update_best_softmax_scores(
                        fitness_score, i, pops, succeeded_pops, success_indicator_index, softmax_numpy[:, label])

                self.total_succeeded_in_iteration[i] = 0
                g_fitness = torch.sum(fitness_score).item()
                p_best_idx = torch.argmin(fitness_score) % len(pops)
                g_fitness_real = torch.min(fitness_score)

                self.pops[i, ...] = pops
                # if diffusion_imgs is not None:
                #     pops = pops + pops * len(diffusion_imgs.generated_imgs_train_cropped_names)
                self.p_best[i] = pops[p_best_idx.item()]
                self.p_best_fitness[i] = fitness_score[p_best_idx.item()]

                if self.best_pop_on_src_and_diffusoin_count_imgs_success == 0:
                    if g_fitness < temp:
                        self.g_best = self.p_best[i]
                        self.g_best_fitness = g_fitness
                        temp = g_fitness

                    if g_fitness_real < temp_real:
                        self.sg_best = self.p_best[i]
                        self.sg_best_fitness = g_fitness_real
                        temp_real = g_fitness_real

                    if self.best_pop_on_src_and_diffusoin_count_imgs_success > 0:
                        print(
                            f"Initialize Success on {filename}-{i}: {self.best_pop_on_src_and_diffusoin_count_imgs_success}, g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                    else:
                        print(
                            f"Initialize Failed {filename}-{i}: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")

                    # print(
                    #     f"Initialize Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness},  p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")

                # self.save_best_general_iteration_result(adv_images[min_index_class], adv_params[min_index_class], self.p_best[i], min_index_class, smallest_score_for_class,
                #                                         smallest_score_attack_succeeded,
                #                                         diffusion_imgs, 0)

        return False

    def find_best_result_for_current_iteration(self, softmax, success_indicator_index, label):
        if len(success_indicator_index) > 0:
            # Select softmax scores for label class  corresponding to incorrect predictions
            class_scores_incorrect = softmax[success_indicator_index.detach().cpu().numpy(), label]
            # Find the index with the smallest softmax score for label class among incorrect predictions
            min_index_class_incorrect = np.argmin(class_scores_incorrect)

            # Get the corresponding index from the incorrect prediction indexes
            min_index_incorrect_predictions = success_indicator_index[min_index_class_incorrect]

            print("Index with smallest softmax score for label class among incorrect predictions:",
                  min_index_incorrect_predictions)
        else:
            # Select softmax scores for label class corresponding to correct predictions (because there ao incorrect predictions)
            class_scores = softmax[:, label]
            # Find the index with the smallest softmax score for label class among incorrect predictions
            min_index_class = np.argmin(class_scores)
            smallest_score_for_class = softmax[min_index_class][label]

    def get_smallest_index_and_score_for_attacking_class(self, softmax, attack_class_index = 12):
        # Select softmax scores for label class corresponding to correct predictions (because there ao incorrect predictions)
        class_scores = softmax[:, attack_class_index]
        # print(f"class_scores.shape: {class_scores.shape}")
        # Find the index with the smallest softmax score for label class among incorrect predictions
        min_index_class = np.argmin(class_scores)
        smallest_score_for_class = softmax[min_index_class][attack_class_index]

        return min_index_class, smallest_score_for_class

    def update_iteration_params(self, fitness_score, i, pops, succeeded_pops, success_indicator_index, softmax_for_atatcked_label_all):
        best_succeeded_pops_ind, best_succeeded_pops_mean_score, mask, total_imgs_in_attack = self.update_best_softmax_scores(
            fitness_score, i, pops, succeeded_pops, success_indicator_index, softmax_for_atatcked_label_all)
        # removing successful imgs
        # We take failed examples and therefore their argmax is == attack_class_label -> so the fitness score will have the score the attacked class
        mask = [score_ind for score_ind in mask if score_ind not in success_indicator_index]
        # Use the mask to select values from the tensor
        selected_values = fitness_score[mask]
        # compute score mean of failed attack of selected pos (the mean is evaluateing considering score for success attack as 0, therefore we divide by total_imgs_in_attack
        self.p_best_fitness[i] = torch.sum(selected_values).item() / total_imgs_in_attack
        # setting best pop
        best_pop = pops[best_succeeded_pops_ind]
        # best_pop = pops[succeeded_pops.argmax()]
        self.p_best[i] = best_pop
        # setting number of attacked successfully images (of src + diffusion imgs)
        return best_succeeded_pops_ind, best_succeeded_pops_mean_score

    def update_best_softmax_scores(self, fitness_score, i, pops, succeeded_pops, success_indicator_index,
                                   softmax_for_atatcked_label_all):

        self.p_best_n_success_imgs[i] = succeeded_pops.max()
        # Calculate mean score for all images with same succeded pop:7
        # Create a mask for getting score for differenet images with same score,
        total_imgs_in_attack = int(len(fitness_score) / len(pops))
        min_index_class, smallest_score_for_attacked_class, best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.find_best_diffusion_index_success(
            pops, succeeded_pops, success_indicator_index, total_imgs_in_attack, softmax_for_atatcked_label_all)
        mask = [int(i * len(pops) + best_succeeded_pops_ind) for i in range(total_imgs_in_attack)]
        # mask = [int(i * len(pops) + succeeded_pops.argmax()) for i in range(total_imgs_in_attack)]
        # Taking softmax scores for all attacked images with same pop
        self.softmax_scores[i] = softmax_for_atatcked_label_all[mask]
        self.is_attack_succeeded_for_each_img[i] = [
            True if score_ind in success_indicator_index else False for score_ind in mask]
        self.lowest_softmax_for_attacked_class[i] = smallest_score_for_attacked_class
        self.lowest_softmax_for_attacked_class_succeeded[i] = min_index_class in success_indicator_index
        self.lowest_softmax_for_attacked_class_index[i] = min_index_class
        self.is_lowest_softmax_for_attacked_class_in_best_chosen_pop[i] = min_index_class % len(pops) == best_succeeded_pops_ind
        self.best_succeeded_pops_mean_scores[i] = best_succeeded_pops_mean_score

        return best_succeeded_pops_ind, best_succeeded_pops_mean_score, mask, total_imgs_in_attack

    def find_best_diffusion_index_success(self, pops, succeeded_pops, success_indicator_index, total_imgs_in_attack,
                                          softmax_for_atatcked_label_all):
        succeeded_n_imgs = succeeded_pops.max()
        min_index_class = np.argmin(softmax_for_atatcked_label_all)
        smallest_score_for_attacked_class = softmax_for_atatcked_label_all[min_index_class]
        pops_len = len(pops)
        min_index_pop = min_index_class % pops_len
        if succeeded_n_imgs == 0:
            pop_indexs = [int(i * pops_len + min_index_pop) for i in range(total_imgs_in_attack)]
            scores = softmax_for_atatcked_label_all[pop_indexs]
            scores_mean = np.mean(scores)
            return min_index_class, smallest_score_for_attacked_class, min_index_pop, scores_mean

        best_succeeded_pops_mean_score = torch.inf
        best_succeeded_pops_ind = -1
        # Find index in succeeded_pops that has the max value and lowest prediction for required attacked class
        for ind in range(len(succeeded_pops)):
            if succeeded_pops[ind] == succeeded_n_imgs:
                # get all indexes of this pop:
                pop_indexs = [int(i * len(pops) + ind) for i in range(total_imgs_in_attack)]
                # scores = fitness_score[pop_indexs]  # TODO: check this, shouldn't it be softmax?
                scores = softmax_for_atatcked_label_all[pop_indexs]  # TODO: check this, shouldn't it be softmax?
                mean_score = np.sum(scores) / total_imgs_in_attack
                if mean_score < best_succeeded_pops_mean_score:
                    best_succeeded_pops_mean_score = mean_score
                    best_succeeded_pops_ind = ind

        return min_index_class, smallest_score_for_attacked_class, best_succeeded_pops_ind, best_succeeded_pops_mean_score

    def compute_best_pop_params_considering_diffusion_imgs(self, fitness_score, pops, success_indicator_index):
        g_fitness_sum = 0
        best_min_fitness_score = torch.inf
        best_min_fitness_score_ind = 0
        for score_ind, score in enumerate(fitness_score):
            if score_ind in success_indicator_index:
                continue
            g_fitness_sum += score
            if fitness_score[score_ind].item() < best_min_fitness_score:
                best_min_fitness_score_ind = score_ind
                best_min_fitness_score = fitness_score[score_ind].item()
        best_min_fitness_score_ind % len(pops)
        g_fitness = g_fitness_sum
        p_best_idx = best_min_fitness_score_ind
        g_fitness_real = best_min_fitness_score

        return g_fitness_sum, best_min_fitness_score_ind, best_min_fitness_score

    # def compute_best_pop_params_with_diffusion_imgs_in_equal_state(self, fitness_score, pops, success_indicator_index):
    #     g_fitness_sum_success, g_fitness_sum_failed = 0
    #     best_success_fitness_score = 0
    #     best_success_fitness_score_ind = 0
    #     for score_ind, score in enumerate(fitness_score):
    #         if score_ind in success_indicator_index:
    #             g_fitness_sum_success += score
    #         else:
    #             g_fitness_sum_failed += score
    #         if fitness_score[score_ind].item() < best_min_fitness_score:
    #             best_min_fitness_score_ind = score_ind
    #             best_min_fitness_score = fitness_score[score_ind].item()
    #     best_min_fitness_score_ind % len(pops)
    #     g_fitness = g_fitness_sum
    #     p_best_idx = best_min_fitness_score_ind
    #     g_fitness_real = best_min_fitness_score
    #
    #     ############
    #     if fitness_score[p_best_idx.item()] < self.p_best_fitness[i]:
    #         self.p_best[i] = self.pops[i][p_best_idx.item()]
    #
    #     if g_fitness < self.g_best_fitness:  # Sum
    #         self.g_best_fitness = g_fitness
    #         self.g_best = self.p_best[i]
    #
    #     if g_fitness_real < self.sg_best_fitness:  # min Prob
    #         self.sg_best_fitness = g_fitness_real
    #         self.sg_best = self.p_best[i]
    #
    #     if itr == self.max_iter - 1 and i == self.size - 1:
    #         current_adv_images_ = cv2.cvtColor(current_adv_images[0], cv2.COLOR_BGR2RGB)
    #         Image.fromarray(current_adv_images_).save(fr'{self.save_dir}/{filename}_failed.png')
    #
    #     print(
    #         f"【{itr}/{self.max_iter}】Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")
    #
    #     ##############33
    #     return g_fitness_sum, best_min_fitness_score_ind, best_min_fitness_score

    def gen_adv_to_multiple_imgs(self, adv_images: np.ndarray, adv_params: List[AttackParams], pops: List[Tuple], diffusion_imgs: DiffusionImages):

        all_diffusion_adv_images_list = [adv_images]
        all_diffusion_adv_params_list = [adv_params]
        for diffusion_img, diffusion_img_name in diffusion_imgs.diffusion_images_for_attack():
            diffusion_adv_images, diffusion_adv_params = self.gen_adv_images_by_pops(diffusion_img, pops)
            all_diffusion_adv_images_list.append(diffusion_adv_images)
            all_diffusion_adv_params_list.append(diffusion_adv_params)
        # adv_images = [img_ for list_ in all_diffusion_adv_images_list for img_ in list_]
        adv_images = np.stack([img_ for list_ in all_diffusion_adv_images_list for img_ in list_], 0)
        adv_params = np.stack([item_ for list_ in all_diffusion_adv_params_list for item_ in list_], 0)

        return adv_images, adv_params

    @torch.no_grad()
    def calculate_fitness(self, images: np.ndarray) -> torch.Tensor:
        if(not isinstance(images, list) and len(images.shape)==3):
            images = self.models_wrapper[0].pre_process_image(images, device=self.device, use_interpolate=self.models_wrapper[0].use_interpolate)
        else:
            resized_images=[]
            num_of_images = len(images) if isinstance(images, list) else images.shape[0]
            for i in range(num_of_images):
                resized_images.append(self.models_wrapper[0].pre_process_image(images[i], device=self.device, use_interpolate=self.models_wrapper[0].use_interpolate))
            resized_images = torch.stack(resized_images, dim=0)
            if len(resized_images.shape) == 5:
                resized_images = resized_images.squeeze(1)
            images = resized_images.to(self.device)

        images = images.to(self.device)

        if len(self.models_wrapper) > 1:
            softmax = self.get_ensemble_prediction(images)
        else:
            output = self.models_wrapper[0].model(images)
            softmax = torch.softmax(output, dim=1)

        return softmax

    def get_ensemble_prediction(self, images: torch.Tensor):
        output_all = []
        for model_index in range(len(self.models_wrapper)):
            output = self.models_wrapper[model_index].model(images)
            softmax = torch.softmax(output, dim=1)
            output_all.append(softmax)

        averaged_outputs = []

        # Loop through each row (dim=0) and average the corresponding tensors
        for row_index in range(output_all[0].shape[0]):
            row_outputs = [output_[row_index, :] for output_ in output_all]
            averaged_row = torch.mean(torch.stack(row_outputs, dim=0), dim=0)
            averaged_outputs.append(averaged_row)

        # Stack the averaged tensors along dim=0 to get the final result
        final_output = torch.stack(averaged_outputs, dim=0)

        return final_output


    def calculate_omega(self, itr: int):
        omega = self.omega_bound[1] - (self.omega_bound[1] - self.omega_bound[0]) * (itr / self.max_iter)
        return omega

    def update(self, orig_image, cropped_image, image: np.ndarray, bbx, label: torch.Tensor, itr: int,
               filename: str = "test_update", diffusion_imgs: DiffusionImages = None):
        c1 = self.c_list[0]
        c2 = self.c_list[1]
        c3 = self.c_list[2]
        w = self.calculate_omega(itr)
        image_raw = copy.deepcopy(image)
        cropped_image_raw = copy.deepcopy(cropped_image)
        orig_image_raw = copy.deepcopy(orig_image)

        num_of_pops_in_each_iteration = self.pops[0].shape[0]
        for i in range(self.size):
            succeeded_pops = np.zeros(len(self.pops[i]))
            ################# Constrain the bound of the circle #################
            circlr_v = w * self.v[i][0, :3] + c1 * random.uniform(0, 1) * (
                    self.p_best[i][:3] - self.pops[i][0, :3]) + c2 * random.uniform(
                0, 1) * (self.g_best[:3] - self.pops[i][0, :3]) + c3 * random.uniform(
                0, 1) * (self.sg_best[:3] - self.pops[i][0, :3])

            for j in range(self.sub_size):
                for k in range(self.dimension):
                    if k > 3:
                        break
                    self.v[i][j][k] = min(max(self.v[i][j][k], self.v_bound[0][k]), self.v_bound[1][k])

            # 更新位置
            self.pops[i][:3] = self.pops[i][:3] + self.v[i][:3]
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    if k > 3:
                        break
                    self.pops[i][j][k] = min(max(self.pops[i][j][k], self.pop_bound[0][k]), self.pop_bound[1][k])

            ################# Finished #################

            ################# Constrain the bound of the reset variables #################
            for j in range(self.sub_size):
                self.v[i][j][:3] = circlr_v
                self.v[i][j][3:] = w * self.v[i][j][3:] + c1 * random.uniform(0, 1) * (
                        self.p_best[i][3:] - self.pops[i][j][3:]) + c2 * random.uniform(
                    0, 1) * (self.g_best[3:] - self.pops[i][j][3:]) + c3 * random.uniform(
                    0, 1) * (self.sg_best[3:] - self.pops[i][j][3:])
            # velocity bound
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    self.v[i][j][k] = min(max(self.v[i][j][k], self.v_bound[0][k]), self.v_bound[1][k])
            # update the solution
            self.pops[i] = self.pops[i] + self.v[i]
            # solution bound
            for j in range(self.sub_size):
                for k in range(self.dimension):
                    self.pops[i][j][k] = min(max(self.pops[i][j][k], self.pop_bound[0][k]), self.pop_bound[1][k])
            ################# Finished!! #################
            current_adv_images, current_adv_params = self.gen_adv_images_by_pops(image_raw, self.pops[i])
            if diffusion_imgs is not None:
                current_adv_images, current_adv_large_images_by_prompt_desc = self.generate_diffusion_images_conatining_attack(current_adv_images, orig_image_raw,
                                                                              cropped_image_raw, bbx, train=True, filename=filename, main_itr=itr, iter=i, initialize=False)
                # current_adv_images, current_adv_params = self.gen_adv_to_multiple_imgs(current_adv_images, current_adv_params, self.pops[i], diffusion_imgs)

            softmax = self.calculate_fitness(current_adv_images)
            fitness_score, pred_labels = torch.max(softmax, dim=1)
            softmax_numpy = copy.deepcopy(softmax).detach().cpu().numpy()

            # If find the best solution, then exist
            success_indicator = (pred_labels != label)

            min_index_class, smallest_score_for_class = self.get_smallest_index_and_score_for_attacking_class(softmax_numpy, label)
            smallest_score_attack_succeeded = pred_labels[min_index_class] != label


            if diffusion_imgs is not None:
                self.add_score_and_indication_if_attack_succeeded_to_img_names(softmax_numpy[:, label],
                                                                  success_indicator, len(self.pops),
                                                                  train=True, filename=filename,
                                                                   main_itr=itr,
                                                                  iter=i, initialize=False)
            success_indicator_index = torch.where(success_indicator)[0]
            # self.find_best_result_for_current_iteration(softmax_numpy, success_indicator_index, label)

            success_indicator_index = success_indicator_index.cpu().data.numpy()
            if len(success_indicator_index) >= 1:
                # if success_indicator.sum().item() >= 1:
                #     image2saved = current_adv_images[success_indicator.cpu().data.numpy()]
                image2saved = current_adv_images[success_indicator_index]
                if diffusion_imgs is not None:
                    for pop_ind in success_indicator_index:
                        # TODO: set 30 again
                        # succeeded_pops[pop_ind % 30] += 1
                        succeeded_pops[pop_ind % len(self.pops)] += 1
                    if succeeded_pops.max() > self.best_pop_on_src_and_diffusoin_count_imgs_success:
                        best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                     success_indicator_index, softmax_numpy[:, label])
                        self.update_best_pop_attack(i, succeeded_pops, best_succeeded_pops_ind, best_succeeded_pops_mean_score, current_adv_large_images_by_prompt_desc)
                        print(f"【{itr}/{self.max_iter}】Success on {succeeded_pops.max()} imgs: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")

                    # if succeeded_pops.max() == (len(diffusion_imgs.generated_imgs_train_cropped_names) + 1):
                    if succeeded_pops.max() == (len(diffusion_imgs.generated_imgs_train_cropped_names) / 2) + 1:
                        best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                     success_indicator_index, softmax_numpy[:, label])
                        self.update_best_pop_attack(i, succeeded_pops, best_succeeded_pops_ind, best_succeeded_pops_mean_score,
                                                    current_adv_large_images_by_prompt_desc)

                        # TODO: apply attack to test generated images...
                        print(f"{filename}: 【{itr}/{self.max_iter}】 i-{i} Success all: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                        return True
                    # elif succeeded_pops.max() == self.best_pop_on_src_and_diffusoin_count_imgs_success:
                    #     g_fitness, p_best_idx, g_fitness_real = self.compute_best_pop_params_with_diffusion_imgs_in_equal_state(
                    #         fitness_score, self.pops[i], success_indicator_index)
                    else:
                        if succeeded_pops.max() > self.p_best_n_success_imgs[i]:
                            best_succeeded_pops_ind, best_succeeded_pops_mean_score = self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                    success_indicator_index, softmax_numpy[:, label])
                else:
                    best_pop = self.pops[i][success_indicator_index[0]]
                    self.best_pop_on_src_and_diffusoin = best_pop
                    # self.p_best[i] = self.pops[i]
                    self.p_best[i] = best_pop
                    self.g_best = self.p_best[i]
                    self.sg_best = self.p_best[i]

                    # image2saved_ = cv2.cvtColor(image2saved[0], cv2.COLOR_BGR2RGB) # TODO: I changed it back to RGB

                    # Image.fromarray(image2saved_).save(fr'{self.save_dir}/{filename}.png')
                    print(
                        f"{filename}: 【{itr}/{self.max_iter}】 i-{i} Success: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                    return True
            else:

                if diffusion_imgs is not None:
                    best_succeeded_pops_ind, best_succeeded_pops_mean_score, mask, total_imgs_in_attack = self.update_best_softmax_scores(
                        fitness_score, i, self.pops[i], succeeded_pops, success_indicator_index, softmax_numpy[:, label])

                g_fitness = torch.sum(fitness_score).item()
                p_best_idx = torch.argmin(fitness_score)
                g_fitness_real = torch.min(fitness_score)

                if self.p_best_n_success_imgs[i] == 0:
                    if fitness_score[p_best_idx.item()] < self.p_best_fitness[i]:
                        self.p_best[i] = self.pops[i][p_best_idx.item() % num_of_pops_in_each_iteration]

                if self.best_pop_on_src_and_diffusoin_count_imgs_success == 0:
                    if g_fitness < self.g_best_fitness:  # Sum
                        self.g_best_fitness = g_fitness
                        self.g_best = self.p_best[i]

                    if g_fitness_real < self.sg_best_fitness: # min Prob
                        self.sg_best_fitness = g_fitness_real
                        self.sg_best = self.p_best[i]

                if itr == self.max_iter-1 and i == self.size-1:
                    current_adv_images_ = cv2.cvtColor(current_adv_images[0], cv2.COLOR_BGR2RGB)
                    Image.fromarray(current_adv_images_).save(fr'{self.save_dir}/{filename}_failed.png') # TODO: check if the save here is needed and if the added 'failed' to the saved image name is needed as well

                if self.best_pop_on_src_and_diffusoin_count_imgs_success > 0:
                    print(
                        f"【{itr}/{self.max_iter}】 i-{i} Success on {filename} i-{i}: {self.best_pop_on_src_and_diffusoin_count_imgs_success}, g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}")
                else:
                    print(f"{filename}:【{itr}/{self.max_iter}】 i-{i} Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")

        return False

    def get_diffusion_score_min_and_sum_of_failed_attack(self, fitness_score, success_indicator_index):
        pass
    def get_diffusion_score_min_and_sum_of_failed_attack(self, fitness_score, success_indicator_index):
        """Iterating for indexes of failed attack, and sum their probs"""
        pop_scores_by_img = torch.zeros(self.pops[0].shape[0])
        size = len(pop_scores_by_img)
        g_fitness_sum_score_failed = 0
        g_fitness_min_score_failed = torch.inf
        g_fitness_min_score_failed_ind = -1
        for score_ind, score in enumerate(fitness_score):
            if score_ind in success_indicator_index:
                continue
            pop_scores_by_img[score_ind % size] += score
            g_fitness_sum_score_failed += score
            # if score < g_fitness_min_score_failed:
            #     g_fitness_min_score_failed = score
            #     g_fitness_min_score_failed_ind = score_ind
        pop_scores_by_img = pop_scores_by_img / (fitness_score / size)
        g_fitness_min_avg_score_failed = torch.min(pop_scores_by_img)
        g_fitness_min_avg_score_failed_ind = torch.argmin(pop_scores_by_img)

        return g_fitness_sum_score_failed, g_fitness_min_avg_score_failed, g_fitness_min_avg_score_failed_ind
    def update_best_pop_attack(self, i, succeeded_pops, best_succeeded_pops_ind, best_succeeded_pops_mean_score, best_diffusion_imgs_by_prompt_desc: Dict[str, PIL.Image.Image]):
        self.best_pop_on_src_and_diffusoin_count_imgs_success = succeeded_pops.max()
        best_pop = self.pops[i][best_succeeded_pops_ind]
        # best_pop = self.pops[i][succeeded_pops.argmax()]
        self.best_pop_on_src_and_diffusoin = best_pop
        self.best_diffusion_imgs_by_prompt_desc = best_diffusion_imgs_by_prompt_desc
        self.best_succeeded_pops_mean_scores[i] = best_succeeded_pops_mean_score
        # self.p_best[i] = best_pop
        self.g_best = self.p_best[i]
        self.sg_best = self.p_best[i]

    def run_pso(self, file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped, attack_with_diffusion: bool): #data_loader

        total = 0
        success_cnt = 0
        size = 224

        for i, (orig_image, cropped_image, image, bbx, mask, filename) in enumerate(zip(orig_imgs, cropped_imgs, cropped_resized_imgs, bbx, masks_cropped, file_names)): #data_loader # iterative each images
            self.mask = mask
            total += 1
            softmax_ori = self.calculate_fitness(image)
            fitness_score_ori, pred_label = torch.max(softmax_ori, dim=1)
            print(f"filename: {filename}.png predicted as: {pred_label.item()}")
            is_find_init = self.initialize(orig_image, cropped_image, image, bbx, pred_label, filename)
            if is_find_init:
                success_cnt += 1
                print("Initial found!!!")
                print("==" * 30)
                continue

            for itr in range(self.max_iter):
                is_find_search = self.update(orig_image, cropped_image, image, bbx, pred_label, itr, filename, None)
                if is_find_search:
                    success_cnt += 1
                    print("==" * 30)
                    break

        asr = round(100 * (success_cnt / total), 2)
        return asr

    def run_pso_with_diffusion_imgs(self, file_names: List[str],
                                    orig_imgs: List[np.ndarray],
                                    cropped_imgs: List[np.ndarray],
                                    cropped_resized_imgs: List[np.ndarray],
                                    labels: List[int],
                                    bbx: List[List[int]],
                                    masks_cropped: List[np.ndarray],
                                    attack_with_diffusion: bool,
                                    stable_diffusion_model: None): #data_loader # diffusers.StableDiffusionPipeline =

        total = 0
        success_cnt = 0
        size = 224

        self.stable_diffusion_main_params = stable_diffusion_xl_main_params()
        self.stable_diffusion_model = stable_diffusion_model
        for i, (orig_img, cropped_img, cropped_resized_img, bbx, mask, filename) in enumerate(zip(orig_imgs, cropped_imgs, cropped_resized_imgs, bbx, masks_cropped, file_names)): #data_loader # iterative each images
            self.reset_attack_params()
            diffusion_imgs = DiffusionImages(filename, bbx, size, self.save_dir)
            self.mask = mask
            total += 1
            softmax_ori = self.calculate_fitness(cropped_resized_img)
            fitness_score_ori, pred_label = torch.max(softmax_ori, dim=1)
            if attack_with_diffusion:
                is_find_init = self.initialize(orig_img, cropped_img, cropped_resized_img, bbx, pred_label, filename,
                                               diffusion_imgs)
                self.save_info(filename, itr=-1)
            else:
                is_find_init = self.initialize(orig_img, cropped_img, cropped_resized_img, bbx, pred_label, filename,
                                               None)

            if is_find_init:
                success_cnt += 1
                print("Initial found!!!")
                print("==" * 30)
                self.save_adv_images(cropped_resized_img, orig_img, cropped_img, bbx, filename, attack_with_diffusion=attack_with_diffusion, diffusion_imgs=diffusion_imgs, main_itr = -1)
                continue

            for itr in range(self.max_iter):
                if attack_with_diffusion:
                    is_find_search = self.update(orig_img, cropped_img, cropped_resized_img, bbx, pred_label, itr,
                                                 filename, diffusion_imgs)
                    self.save_info(filename, itr=itr)

                else:
                    is_find_search = self.update(orig_img, cropped_img, cropped_resized_img, bbx, pred_label, itr,
                                                 filename, None)

                if is_find_search:
                    success_cnt += 1
                    print("==" * 30)
                    break

            self.save_adv_images(cropped_resized_img, orig_img, cropped_img, bbx, filename, attack_with_diffusion=attack_with_diffusion, diffusion_imgs=diffusion_imgs, main_itr=itr)

            if i==1:
                break
            # # TODO: remove this break
            # if i==1:
            #     break

        asr = round(100 * (success_cnt / total), 2)

        return asr

    def save_adv_images(self, image: np.ndarray, orig_image, orig_cropped_image, bbx, filename: str, attack_with_diffusion: bool = False, diffusion_imgs: DiffusionImages = None, main_itr = None):
        filename_without_ext = filename.split('.')[0]
        best_pop = self.best_pop_on_src_and_diffusoin
        if best_pop is None:
            # best_pop = None #TODO: set best pop
            # Choose it better
            best_pop = self.pops[0][0] #TODO: set some other better best pop
        adv_img, adv_params = self.gen_adv_images_by_pops(image, [best_pop])
        if attack_with_diffusion and not diffusion_imgs:
            raise Exception("attack_with_diffusion is True but diffusion_imgs is None ")

        image2saved_ = cv2.cvtColor(adv_img[0], cv2.COLOR_BGR2RGB)  # TODO: check is really needed
        if not diffusion_imgs:
            Image.fromarray(image2saved_).save(fr'{self.save_dir}/{filename}.png')
        else:
            if attack_with_diffusion:
                output_dir = diffusion_imgs.output_dir_special
                output_dir_large = os.path.join(output_dir, 'large')
                os.makedirs(output_dir_large, exist_ok=True)
            else:
                output_dir = diffusion_imgs.output_dir_normal
            print("output_dir: ", output_dir)
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(image2saved_).save(f'{output_dir}/{filename}.png')
            if attack_with_diffusion:
                _, diffusion_adv_large_images_by_prompt_desc = self.generate_diffusion_images_conatining_attack(np.expand_dims(adv_img[0], axis=0),
                                                                                                               orig_image,
                                                                                                               orig_cropped_image,
                                                                                                               bbx,
                                                                                                               train=False,
                                                                                                               filename=filename,
                                                                                                               main_itr=main_itr,
                                                                                                               iter=-1,
                                                                                                               initialize=False)
                for ind, diffusion_adv_data in diffusion_adv_large_images_by_prompt_desc.items():
                    for prompt_desc in diffusion_adv_data.keys():
                        if prompt_desc in diffusion_adv_data:
                            large_adv_img = diffusion_adv_data[prompt_desc]["large_adv_image"]
                            cropped_resize_img = diffusion_adv_data[prompt_desc]["cropped_resize_img"]

                            # adv_img, adv_params = self.gen_adv_images_by_pops(diffusion_img, [best_pop])
                            image2saved_ = cv2.cvtColor(cropped_resize_img, cv2.COLOR_BGR2RGB)  # TODO: check is really needed
                            Image.fromarray(image2saved_).save(fr'{output_dir}/{prompt_desc}.png')

                            # Saving large adv image
                            # adv_img, adv_params = self.gen_adv_images_by_pops(diffusion_img, [best_pop])
                            image2saved_ = cv2.cvtColor(np.array(large_adv_img), cv2.COLOR_BGR2RGB)  # TODO: check is really needed
                            Image.fromarray(image2saved_).save(fr'{output_dir_large}/{prompt_desc}.png')

                            adv_params_pickle_file_path = fr'{output_dir}/{prompt_desc}.pkl'
                            with open(adv_params_pickle_file_path, 'wb') as file:
                                pickle.dump(adv_params[0], file)
                            image_attack_mask_path =fr'{output_dir}/{prompt_desc}_mask.png'
                            mask_attack = np.ones_like(image) * 255
                            cv2.fillPoly(mask_attack, adv_params[0].points, (0, 0, 0))
                            cv2.imwrite(image_attack_mask_path, mask_attack)
                        else:
                            print(f"Warning: prompt_desc `{prompt_desc}` not in diffusion_adv_data")

            # for diffusion_img, diffusion_img_name in diffusion_imgs.diffusion_images_for_test():
            #     adv_img, adv_params = self.gen_adv_images_by_pops(diffusion_img, [best_pop])
            #     image2saved_ = cv2.cvtColor(adv_img[0], cv2.COLOR_BGR2RGB)  # TODO: check is really needed
            #     Image.fromarray(image2saved_).save(fr'{output_dir}/{diffusion_img_name}.png')
            #     adv_params_pickle_file_path = fr'{output_dir}/{diffusion_img_name}.pkl'
            #     with open(adv_params_pickle_file_path, 'wb') as file:
            #         pickle.dump(adv_params[0], file)
            #     image_attack_mask_path =fr'{output_dir}/{diffusion_img_name}_mask.png'
            #     mask_attack = np.ones_like(diffusion_img) * 255
            #     cv2.fillPoly(mask_attack, adv_params[0].points, (0, 0, 0))
            #     cv2.imwrite(image_attack_mask_path, mask_attack)

    @timeit
    def generate_diffusion_images_conatining_attack(self, adv_images, src_orig_image, src_cropped_image, bbx,
                                                    train: bool = True, filename: str = None, main_itr=None, iter=None, initialize: bool = True):
        sub_dir = 'init' if initialize else 'non_init'
        current_logs_dir = os.path.join(self.logs_dir, filename, sub_dir, str(main_itr), str(iter))
        os.makedirs(current_logs_dir, exist_ok=True)

        for adv_i, adv_img in enumerate(adv_images):
            adv_img_out_name = f'init-{int(initialize)}_mainItr-{main_itr}_iter-{iter}_adv-{adv_i}.png'
            cv2.imwrite(os.path.join(current_logs_dir, adv_img_out_name), adv_images[adv_i])

        prompt_types = GENERATED_IMAGES_TYPES_TRAIN if train else GENERATED_IMAGES_TYPES_TEST
        adv_images_large = self.insert_cropped_resized_with_attack_into_large_image(adv_images, src_orig_image,
                                                                                   src_cropped_image, bbx)
        # all_diffusion_adv_images_info = {0:{"src_adv": {"ind+i":0, "cropped_resize_img": adv_images, "large_adv_image": adv_images_large}}}  #[adv_images]
        all_diffusion_adv_images_info = {}
        if adv_images.shape[0] == 1:
            all_diffusion_adv_images_info = {-1:{"src_adv": {"i":-1, "cropped_resize_img": adv_images[0], "large_adv_image": adv_images_large}}}  #[adv_images]
        all_diffusion_adv_images_list = [adv_img for adv_img in adv_images]  #[adv_images]
        adv_images_large_pil = self.numpy_to_pil_img_for_diffusion_model(adv_images_large)
        # print(f"adv_images_large_pil: {len(adv_images_large_pil)}")
        ind = 15
        cnt = 0
        for prompt_desc, cur_prompt in prompt_getter.items():
            if prompt_desc in prompt_types:
                for i, adv_image_large_pil in enumerate(adv_images_large_pil):
                    if self.stable_diffusion_model is not None:
                        # result_image = self.stable_diffusion_model(prompt=cur_prompt + ". do not change the hexagon shadow in the middle of the stop sign",
                        model_type, result_image = self.run_stable_diffusion(adv_image_large_pil, cur_prompt)
                    else:
                        model_type, result_image = self.run_stable_diffusion_using_api(adv_image_large_pil, cur_prompt,
                                                                                       model_type, prompt_desc,
                                                                                       result_image)

                    img_out_name = f'init-{int(initialize)}_{model_type}_mainItr-{main_itr}_iter-{iter}_{prompt_desc}_adv-{i}.png'
                    result_image_resized_cropped = self.process_image_after_diffusion_generation(copy.deepcopy(result_image), bbx, adv_images.shape[1:3])
                    cv2.imwrite(os.path.join(current_logs_dir, img_out_name), result_image_resized_cropped)
                    if adv_images.shape[0] == 1:
                        if cnt not in all_diffusion_adv_images_info:
                            all_diffusion_adv_images_info[cnt] = {}
                        all_diffusion_adv_images_info[cnt][prompt_desc] = {"cnt": cnt, "cropped_resize_img": result_image_resized_cropped, "large_adv_image": copy.deepcopy(result_image)}

                    # if ind + i not in all_diffusion_adv_images_info:
                    #     all_diffusion_adv_images_info[ind + i] = {}
                    # # all_diffusion_adv_images_info[ind + i][prompt_desc] = {"ind+i": ind+i, "cropped_resize_img": result_image_resized_cropped, "large_adv_image": result_image}
                    # all_diffusion_adv_images_info = {}
                    all_diffusion_adv_images_list.append(result_image_resized_cropped)

                ind += 15
                cnt += 1
        adv_images = np.stack(all_diffusion_adv_images_list, 0)
        # adv_images = np.stack([img_ for list_ in all_diffusion_adv_images_list for img_ in list_], 0)

        return adv_images, all_diffusion_adv_images_info

    def add_score_and_indication_if_attack_succeeded_to_img_names(self, softmax_scores_for_class,
                                                                  success_indicator, pops_len,
                                                                  train: bool = True, filename: str = None,
                                                                  main_itr=None, iter=None, initialize: bool = True):
        sub_dir = 'init' if initialize else 'non_init'
        current_logs_dir = os.path.join(self.logs_dir, filename, sub_dir, str(main_itr), str(iter))

        for adv_i in range(pops_len):
            adv_img_out_name = f'init-{int(initialize)}_mainItr-{main_itr}_iter-{iter}_adv-{adv_i}.png'
            new_name = '.'.join(adv_img_out_name.split('.')[:-1]) + f'_score-{softmax_scores_for_class[adv_i].item():.2f}_attackSuccess-{int(success_indicator[adv_i].item()):.2f}.png'
            # print(f"file exist: {os.path.exists(os.path.join(current_logs_dir, adv_img_out_name))}")
            os.rename(os.path.join(current_logs_dir, adv_img_out_name), os.path.join(current_logs_dir, new_name))

        prompt_types = GENERATED_IMAGES_TYPES_TRAIN if train else GENERATED_IMAGES_TYPES_TEST

        model_type = 'local' if self.stable_diffusion_model is not None else 'api'
        prompt_ind = 1
        for prompt_desc, cur_prompt in prompt_getter.items():
            if prompt_desc in prompt_types:
                for ind in range(pops_len):
                    adv_i = prompt_ind * pops_len + ind
                    img_out_name = f'init-{int(initialize)}_{model_type}_mainItr-{main_itr}_iter-{iter}_{prompt_desc}_adv-{ind}.png'
                    new_name = '.'.join(img_out_name.split('.')[:-1]) + f'_score-{softmax_scores_for_class[adv_i].item():.2f}_attackSuccess-{int(success_indicator[adv_i].item()):.2f}.png'
                    # print(f"file exist: {os.path.exists(os.path.join(current_logs_dir, img_out_name))}")
                    os.rename(os.path.join(current_logs_dir, img_out_name), os.path.join(current_logs_dir, new_name))

                prompt_ind += 1



    @timeit
    def run_stable_diffusion_using_api(self, adv_image_large_pil, cur_prompt, model_type, prompt_desc, result_image):
        result_image = self.generate_diffusion_image_using_api(prompt_desc=prompt_desc,
                                                               prompt=cur_prompt + ". do not change the hexagon shadow in the middle of the stop sign",
                                                               image=adv_image_large_pil,
                                                               negative_prompt=NEGATIVE_PROMPT,
                                                               params=self.stable_diffusion_main_params)
        model_type = 'api'
        return model_type, result_image

    @timeit
    def run_stable_diffusion(self, adv_image_large_pil, cur_prompt):
        reinitialize_generator(self.generator, self.generator_seed)
        result_image = self.stable_diffusion_model(
            prompt=cur_prompt + ". do not change the rectangle shadow inside the stop sign",
            # prompt=cur_prompt + ". do not change the rectangle shadow in the middle of the stop sign",
            generator=self.generator,
            image=adv_image_large_pil,
            negative_prompt=NEGATIVE_PROMPT,
            **self.stable_diffusion_main_params)
        model_type = 'local'
        result_image = result_image.images[0]
        return model_type, result_image

    @staticmethod
    def numpy_to_pil_img_for_diffusion_model(numpy_images):
        pil_images = []
        for numpy_image in numpy_images:
            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(numpy_image)

            # Apply EXIF transpose
            pil_image = ImageOps.exif_transpose(pil_image)

            # Convert to RGB mode
            pil_image = pil_image.convert("RGB")

            pil_images.append(pil_image)

        return pil_images

    @staticmethod
    def insert_cropped_resized_with_attack_into_large_image(adv_images, src_orig_image, src_cropped_image, bbx):
        all_adv_large_images = []
        for ind in range(adv_images.shape[0]):
            adv_large = copy.deepcopy(src_orig_image)
            # print(f"ind: {ind}, adv_large.shape: {adv_large.shape}, adv_images[ind].shape: {adv_images[ind].shape}\n src_cropped_image.shape: {src_cropped_image.shape}, src_cropped_image.shape[:2][::-1]: {src_cropped_image.shape[:2][::-1]}")
            adv_image = cv2.resize(adv_images[ind], src_cropped_image.shape[:2][::-1])
            xmin, ymin, xmax, ymax = bbx
            adv_large[ymin:ymax, xmin:xmax] = adv_image
            all_adv_large_images.append(adv_large)

        return all_adv_large_images

    @staticmethod
    def process_image_after_diffusion_generation(result_image, bbx, new_shape):
        img = np.array(result_image)
        xmin, ymin, xmax, ymax = bbx
        img_cropped = crop_image(img, xmin, ymin, xmax, ymax)
        img_cropped_resized = cv2.resize(img_cropped, new_shape)

        return img_cropped_resized

    @staticmethod
    def generate_diffusion_image_using_api(prompt_desc, prompt, image, negative_prompt, params):
        breaker = False
        max_retries = 5
        retry_count = 0
        total_cost = 0
        data = {
            "model": "stable-diffusion-v1-5",
            "prompt": "a photo of an astronaut riding a horse on mars",
            "negative_prompt": "",
            "image": "...looooong base64 encoded image string...",
            "strength": 0.7,
            "steps": 30,
            "guidance": 20,
            "seed": 42,
            "scheduler": "euler",
            "output_format": "jpeg"
        }
        headers = {
            "Authorization": "Bearer Your_Auth_Token",
            "Content-Type": "application/json"
        }
        URL_BASE = 'https://api.getimg.ai/v1'
        URL_SUFFIX = '/stable-diffusion-xl/image-to-image'
        url = URL_BASE + URL_SUFFIX

        while retry_count < max_retries:
            seed = 525901257    #random.randint(1, 2147483647)
            print(f"seed: {seed}")
            bearer_token = TOKENS[6]

            # convert the image to base64 encoding
            _, buffer = cv2.imencode('.jpg', np.array(image))
            base64_image = base64.b64encode(buffer).decode('utf-8')

            data['model'] = 'stable-diffusion-xl-v1-0'
            data['image'] = base64_image
            data['prompt'] = prompt
            data['negative_prompt'] = negative_prompt
            data['strength'] = params['strength']
            data['steps'] = params['num_inference_steps']
            data['guidance'] = params['guidance_scale']
            data['seed'] = seed
            headers['Authorization'] = 'Bearer ' + bearer_token

            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                response_json = response.json()

                # Extract the base64 encoded image string from the JSON response
                base64_image_string = response_json["image"]

                # decode the base64 image string to bytes
                image_bytes = base64.b64decode(base64_image_string)

                # convert the image bytes to a np array and decode into an OpenCV image
                image_array = np.frombuffer(image_bytes, np.uint8)
                cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                total_cost += response_json['cost']
                print('Current cost:', response_json['cost'], 'Total cost:', total_cost)

                return  cv_image
            else:
                retry_count += 1
                error_json = json.loads(response.text)
                print("Request failed:", error_json['error']['code'])
                if error_json['error']['code'] == 'quota_exceeded':
                    print('Trying the next token...')
                    TOKENS.pop(0)
                if len(TOKENS) == 0:
                    print('No more tokens to try, exiting...')
                    breaker = True
                    break
            if breaker:
                break

        return None

    def save_info(self, filename: str, itr: int):
        # Saving data from different self params from reset_attack_params that has info on each inner iteration
        # and each inner iteration will be a line in a dataframe, and in each itr we append to the file
        path_pkl = os.path.join(self.logs_dir, f'iterations_info.pkl')
        path_csv = os.path.join(self.logs_dir, f'iterations_info.csv')

        # Creating a dictionary to store information for each iteration
        iteration_data = {
            "fname": filename,
            "main_itr": itr,
            'i': list(np.arange(len(self.lowest_softmax_for_attacked_class_index))),
            "total_succeeded_in_iteration": self.total_succeeded_in_iteration,
            'p_best_fitness': [float(x.detach().cpu().numpy()) if isinstance(x, torch.Tensor) else x for x in self.p_best_fitness],
            'best_succeeded_pops_mean_scores': [np.round(x, 3) for x in self.best_succeeded_pops_mean_scores],
            'softmax_scores': self.softmax_scores,
            'is_attack_succeeded_for_each_img': self.is_attack_succeeded_for_each_img,
            'lowest_softmax_for_attacked_class': [np.round(x, 3) for x in self.lowest_softmax_for_attacked_class],
            'lowest_softmax_for_attacked_class_index': self.lowest_softmax_for_attacked_class_index,
            'lowest_softmax_for_attacked_class_succeeded': self.lowest_softmax_for_attacked_class_succeeded,
            'is_lowest_softmax_for_attacked_class_in_best_chosen_pop': self.is_lowest_softmax_for_attacked_class_in_best_chosen_pop,
            'p_best': [x[1] for x in self.p_best],
            # 'g_best': [x.item() for x in self.g_best]
        }

        # is_attack_succeeded_for_lowst_softmax_for_attacked_class

        # Convert the dictionary to a DataFrame
        iteration_df = pd.DataFrame(iteration_data)

        self.save_iteration_info_to_pickle(iteration_df, path_pkl)

        self.save_iteration_info_to_csv(iteration_df, path_csv)

    def save_iteration_info_to_csv(self, iteration_df, path_csv):
        # Save to CSV
        if not os.path.exists(path_csv):
            # If the file doesn't exist, simply save the DataFrame to it
            iteration_df.to_csv(path_csv, index=False)
        else:
            # If the file exists, append the new data to it
            iteration_df.to_csv(path_csv, mode='a', header=False, index=False)

    def save_iteration_info_to_pickle(self, iteration_df, path_pkl):
        # Appending the DataFrame to the file
        if not os.path.exists(path_pkl):
            # If file doesn't exist, create a new file and save the DataFrame
            iteration_df.to_pickle(path_pkl)
        else:
            # If file exists, append the DataFrame to the existing file
            existing_df = pd.read_pickle(path_pkl)
            updated_df = pd.concat([existing_df, iteration_df], ignore_index=True)
            updated_df.to_pickle(path_pkl)


def set_bounds(args: argparse.ArgumentParser) -> List[float]:
    if "line" in args.shape_type:
        args.alpha_bound = [0, 1]
        lower_bound = [args.x_bound[0], args.y_bound[0], args.radius_bound[0], args.alpha_bound[0], args.color_bound[0],
                       args.color_bound[0], args.color_bound[0], args.angle_boud[0]]
        higher_bound = [args.x_bound[1], args.y_bound[1], args.image_size, args.alpha_bound[1], args.color_bound[1],
                        args.color_bound[1], args.color_bound[1], args.angle_boud[1]]
    elif "rectangle" in args.shape_type or "triangle" in args.shape_type:
        lower_bound = [args.x_bound[0], args.y_bound[0], args.radius_bound[0], args.alpha_bound[0], args.color_bound[0],
                       args.color_bound[0], args.color_bound[0], args.angle_boud[0], args.angle_boud[0]]
        higher_bound = [args.x_bound[1], args.y_bound[1], args.image_size * 0.4, args.alpha_bound[1],
                        args.color_bound[1], args.color_bound[1], args.color_bound[1], args.angle_boud[1],
                        args.angle_boud[1]]
    else:
        lower_bound = [args.x_bound[0], args.y_bound[0], args.radius_bound[0], args.alpha_bound[0], args.color_bound[0],
                       args.color_bound[0], args.color_bound[0], args.angle_boud[0], args.angle_boud[0],
                       args.angle_boud[0]]
        higher_bound = [args.x_bound[1], args.y_bound[1], args.image_size * 0.4, args.alpha_bound[1],
                        args.color_bound[1], args.color_bound[1], args.color_bound[1], args.angle_boud[1],
                        args.angle_boud[1], args.angle_boud[1]]

    return lower_bound, higher_bound

def load_wrapper_model_and_attacked_label(args: argparse.ArgumentParser):
    """Loading model to attack and orig label.
    loaded model will be set as args.model_wrapper
    orig label will be set as args.attack_label"""

    if args.model_name == LISA:
        if args.attack_label is None or args.attack_label == 'None':
            args.attack_label = STOP_SIGN_LISA_LABEL
        args.model_wrapper = LisaModel(args.is_adv_model, args.image_size_feed_to_model, args.use_interpolate)
        if args.ensemble:
            args.model_wrapper = [args.model_wrapper, LisaModel(not args.is_adv_model, args.image_size_feed_to_model, args.use_interpolate)]
        else:
            args.model_wrapper = [args.model_wrapper]

    else:
        if args.attack_label is None or args.attack_label == 'None':
            args.attack_label = STOP_SIGN_GTSRB_LABEL
        args.model_wrapper = GtsrbModel(args.is_adv_model, args.image_size_feed_to_model, args.use_interpolate)
        if args.ensemble:
            args.model_wrapper = [args.model_wrapper, GtsrbModel(not args.is_adv_model, args.image_size_feed_to_model, args.use_interpolate)]
        else:
            args.model_wrapper = [args.model_wrapper]


def loading_config_file(parser: argparse.ArgumentParser):
    known_args, remaining = parser.parse_known_args()
    with open(known_args.yaml_file, 'r', encoding="utf-8") as fr:
        yaml_file = yaml.safe_load(fr)
        parser.set_defaults(**yaml_file)
    args = parser.parse_args(remaining)
    print(args)

    return args

def main():
    print("Starting...")
    parser = argparse.ArgumentParser(description="Random Search Parameters")
    ################# the file path of config.yml #################
    parser.add_argument("--yaml_file", type=str, default="attacks/RFLA/config.yml", help="the settings config")
    ################# load config.yml file    ##################
    args = loading_config_file(parser)

    # Load wrapper model - results from function are added to args.
    load_wrapper_model_and_attacked_label(args)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    batch_size = args.batch_size
    device = DEVICE

    # Loading data
    assert args.dataset_name == LARGER_IMAGES
    input_data = InputData(args.dataset_name)
    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped = process_image(
        input_data.input_folder,
        input_data.annotations_folder,
        args.model_name, crop_size=args.image_size, mask_folder=input_data.mask_folder)

    # Set output dir
    experiment_dir = os.path.join(args.output_dir, input_data.input_name.lower(),
                                  f'physical_attack_RFLA_{args.model_name}_isAdv-{int(args.is_adv_model)}_shape-{args.shape_type}_maxIter-{args.max_iter}_ensemble-{int(args.ensemble)}_interploate-{int(args.use_interpolate)}')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # r, x, y, alpha, red, green, blue, angle, angle
    lower_bound, higher_bound = set_bounds(args)
    v_higher = np.array([5, 5, 10, 0.05, 5, 5, 5, 10, 10, 10])
    v_lower = -np.array(v_higher)
    c_list = [args.c1, args.c2, args.c3]
    pso = PSOAttack(mask=None,
                    model_wrapper=args.model_wrapper,
                    # pre_process=pre_process,
                    image_size=args.image_size,
                    dimension=args.dimension,
                    max_iter=args.max_iter,
                    size=args.pop_size,
                    sub_size=args.sub_size,
                    c_list=c_list,
                    omega_bound=args.omega_bound,
                    lower_bound=lower_bound,
                    higher_bound=higher_bound,
                    v_lower=v_lower,
                    v_higher=v_higher,
                    shape_type=args.shape_type,
                    save_dir=experiment_dir,
                    logs_dir=log_dir,
                    device=device,
                    generator_seed=args.generator_seed,
                    generator = get_generator(args.generator_seed) if args.diffusion_model_local else None
                    )

    # asr = pso.run_pso(file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped)

    stable_diffusion_model = load_stable_diffusion_xl() if args.diffusion_model_local else None
    asr_without_diffusion = pso.run_pso_with_diffusion_imgs(file_names, orig_imgs, cropped_imgs, cropped_resized_imgs,
                                                            labels, bbx, masks_cropped, attack_with_diffusion=False,
                                                            stable_diffusion_model=None)
    print(f"ASR without diffusion of {args.dataset_name}_{args.model_name} is: {asr_without_diffusion}")

    asr_with_diffusion = pso.run_pso_with_diffusion_imgs(file_names, orig_imgs, cropped_imgs, cropped_resized_imgs,
                                                         labels, bbx, masks_cropped, attack_with_diffusion=True,
                                                         stable_diffusion_model=stable_diffusion_model)
    print(f"ASR with diffusion of {args.dataset_name}_{args.model_name} is: {asr_with_diffusion}")

    inference_on_src_attacked.main(args.model_wrapper[0].model_name, experiment_folder=experiment_dir,
                                   attack_methods=[ATTACK_TYPE_A, ATTACK_TYPE_B], save_results=True,
                                   is_adv_model=args.is_adv_model)

    if args.ensemble:
        # Currently we have at most 2 models in the ensemble
        inference_on_src_attacked.main(args.model_wrapper[1].model_name, experiment_folder=experiment_dir,
                                       attack_methods=[ATTACK_TYPE_A, ATTACK_TYPE_B], save_results=True,
                                       save_to_file_type='a', is_adv_model=not args.is_adv_model)

    if args.plot_pairs:
        create_pair_plots(experiment_dir)
    print("Finished !!!")


# if __name__ == "__main__":
    # main()