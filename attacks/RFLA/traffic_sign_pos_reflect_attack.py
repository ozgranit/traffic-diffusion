import argparse
from typing import List, Tuple

import yaml
import time
import utils_rfla
from buildData.diffusion_generated_imgs import DiffusionImages
from buildData.input_data import InputData
from datasets.larger_images.larger_images_settings import LARGER_IMAGES
from models.gtsrb_model import GtsrbModel
from models.lisa_model import LisaModel
from models.base_model import BaseModel
from inferenceAndResults import inference_on_src_attacked
from load_images import process_image
from plot_images import create_pair_plots
from settings import ATTACK_TYPE_A, ATTACK_TYPE_B, LISA, GTSRB, STOP_SIGN_LISA_LABEL, STOP_SIGN_GTSRB_LABEL, DEVICE
from utils_rfla import *
import random

# Set seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Set this to False for fully deterministic results
torch.use_deterministic_algorithms(True)

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
                 save_dir='saved_images'
                 ):
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
        self.models_wrapper = model_wrapper  # the model
        # self.model_name = model_name
        self.transform = transform  # transformation for input of the model
        # self.pre_process = pre_process
        self.save_dir = save_dir

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

    def gen_adv_images_by_pops(self, image: np.ndarray, pops: List[Tuple]) -> np.ndarray:
        result_images = []
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
                x_11, y_11 = get_symmetric_point_of_center((x, y), (x_0, y_0))
                points = np.array([(x_0, y_0), (x_1, y_1), (x_11, y_11)]).astype(np.int32)
                cv2.fillPoly(image_raw, [points], (red, green, blue))
                image_new = cv2.addWeighted(image_raw, alpha, image, 1 - alpha, 0)

            elif self.shape_type in "rectangle":
                angle_beta = angles[1]
                x_1, y_1 = utils_rfla.get_point_by_angle((x, y), r, angle_beta)
                x_11, y_11 = get_symmetric_point_of_center((x, y), (x_0, y_0))
                x_21, y_21 = get_symmetric_point_of_center((x, y), (x_1, y_1))
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
        return np.array(result_images)

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

    def initialize(self, image: np.ndarray, label: torch.Tensor, filename: str = "test", diffusion_imgs: DiffusionImages = None):
        image_raw = copy.deepcopy(image)
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

            adv_images = self.gen_adv_images_by_pops(image_raw, pops)
            if diffusion_imgs is not None:
                adv_images = self.gen_adv_to_multiple_imgs(adv_images, pops, diffusion_imgs)

            softmax = self.calculate_fitness(adv_images)
            fitness_score, pred_labels = torch.max(softmax, dim=1)

            # Exit when the best solution is obtained
            success_indicator = (pred_labels != label)
            # success_indicator_ = success_indicator.cpu().data.numpy()
            success_indicator_index = torch.where(success_indicator)[0]
            success_indicator_index = success_indicator_index.cpu().data.numpy()
            if len(success_indicator_index) >= 1:
                # if success_indicator.sum().item() >= 1:
                # adv_images[(pred_labels != label).cpu().data.numpy()]
                image2saved = adv_images[success_indicator_index]
                if diffusion_imgs is not None:
                    for pop_ind in success_indicator_index:
                        succeeded_pops[pop_ind % 30] += 1

                    if succeeded_pops.max() > self.best_pop_on_src_and_diffusoin_count_imgs_success:
                        self.update_iteration_params(fitness_score, i, pops, succeeded_pops,
                                                     success_indicator_index)
                        self.update_best_pop_attack(i, succeeded_pops)
                        print(
                            f"Initialize Success on: : {self.best_pop_on_src_and_diffusoin_count_imgs_success}, g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")

                    if succeeded_pops.max() == (len(diffusion_imgs.generated_imgs_train_cropped_names) + 1):
                        self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                     success_indicator_index)
                        self.update_best_pop_attack(i, succeeded_pops)

                        # TODO: apply attack to test generated images...
                        print(
                            f"Initialize Success All: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
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

                        self.update_iteration_params(fitness_score, i, pops, succeeded_pops, success_indicator_index)

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
                    best_pop = self.pops[i][success_indicator_index[0]]
                    self.best_pop_on_src_and_diffusoin = best_pop
                    self.p_best[i] = best_pop
                    self.g_best = self.p_best[i]
                    self.sg_best = self.p_best[i]

                    # image2saved_ = cv2.cvtColor(image2saved[0], cv2.COLOR_BGR2RGB) # TODO: check is really needed
                    # Image.fromarray(image2saved_).save(fr'{self.save_dir}/{filename}.png')
                    print(
                        f"Initialize Success: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                    return True
            else:
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
                            f"Initialize Success on: {self.best_pop_on_src_and_diffusoin_count_imgs_success}, g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                    else:
                        print(
                            f"Initialize Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")

                    # print(
                    #     f"Initialize Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness},  p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")

        return False

    def update_iteration_params(self, fitness_score, i, pops, succeeded_pops, success_indicator_index):
        self.p_best_n_success_imgs[i] = succeeded_pops.max()
        # Calculate mean score for all images with same succeded pop:7
        # Create a mask for getting score for differenet images with same score,
        total_imgs_in_attack = int(len(fitness_score) / len(pops))
        mask = [int(i * len(pops) + succeeded_pops.argmax()) for i in range(total_imgs_in_attack)]
        # removing successed imgs
        mask = [score_ind for score_ind in mask if score_ind not in success_indicator_index]
        # Use the mask to select values from the tensor
        selected_values = fitness_score[mask]
        # compute score mean of failed attack of selected pos (the mean is evaluateing considering score for success attack as 0, therefore we divide by total_imgs_in_attack
        self.p_best_fitness[i] = torch.sum(selected_values).item() / total_imgs_in_attack
        # setting best pop
        best_pop = pops[succeeded_pops.argmax()]
        self.p_best[i] = best_pop

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

    def gen_adv_to_multiple_imgs(self, adv_images: np.ndarray, pops: List[Tuple], diffusion_imgs: DiffusionImages):

        all_diffusion_adv_images_list = [adv_images]
        for diffusion_img, diffusion_img_name in diffusion_imgs.diffusion_images_for_attack():
            diffusion_adv_images = self.gen_adv_images_by_pops(diffusion_img, pops)
            all_diffusion_adv_images_list.append(diffusion_adv_images)
        # adv_images = [img_ for list_ in all_diffusion_adv_images_list for img_ in list_]
        adv_images = np.stack([img_ for list_ in all_diffusion_adv_images_list for img_ in list_], 0)

        return adv_images

    @torch.no_grad()
    def calculate_fitness(self, images: np.ndarray) -> torch.Tensor:
        if(not isinstance(images, list) and len(images.shape)==3):
            images = self.models_wrapper[0].pre_process_image(images, device=DEVICE, use_interpolate=self.models_wrapper[0].use_interpolate)
        else:
            resized_images=[]
            num_of_images = len(images) if isinstance(images, list) else images.shape[0]
            for i in range(num_of_images):
                resized_images.append(self.models_wrapper[0].pre_process_image(images[i], device=DEVICE, use_interpolate=self.models_wrapper[0].use_interpolate))
            resized_images = torch.stack(resized_images, dim=0)
            if len(resized_images.shape) == 5:
                resized_images = resized_images.squeeze(1)
            images = resized_images.to(device)

        images = images.to(device)

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

    def update(self, image: np.ndarray, label: torch.Tensor, itr: int, filename: str = "test_update", diffusion_imgs: DiffusionImages = None):
        c1 = c_list[0]
        c2 = c_list[1]
        c3 = c_list[2]
        w = self.calculate_omega(itr)
        image_raw = copy.deepcopy(image)
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
            current_adv_images = self.gen_adv_images_by_pops(image_raw, self.pops[i])
            if diffusion_imgs is not None:
                current_adv_images = self.gen_adv_to_multiple_imgs(current_adv_images, self.pops[i], diffusion_imgs)

            softmax = self.calculate_fitness(current_adv_images)
            fitness_score, pred_labels = torch.max(softmax, dim=1)

            # If find the best solution, then exist
            success_indicator = (pred_labels != label)
            success_indicator_index = torch.where(success_indicator)[0]
            success_indicator_index = success_indicator_index.cpu().data.numpy()
            if len(success_indicator_index) >= 1:
                # if success_indicator.sum().item() >= 1:
                #     image2saved = current_adv_images[success_indicator.cpu().data.numpy()]
                image2saved = current_adv_images[success_indicator_index]
                if diffusion_imgs is not None:
                    for pop_ind in success_indicator_index:
                        succeeded_pops[pop_ind % 30] += 1
                    if succeeded_pops.max() > self.best_pop_on_src_and_diffusoin_count_imgs_success:
                        self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                     success_indicator_index)
                        self.update_best_pop_attack(i, succeeded_pops)
                        print(f"【{itr}/{self.max_iter}】Success: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")

                    if succeeded_pops.max() == (len(diffusion_imgs.generated_imgs_train_cropped_names) + 1):
                        self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                     success_indicator_index)
                        self.update_best_pop_attack(i, succeeded_pops)

                        # TODO: apply attack to test generated images...
                        print(f"【{itr}/{self.max_iter}】Success: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                        return True
                    # elif succeeded_pops.max() == self.best_pop_on_src_and_diffusoin_count_imgs_success:
                    #     g_fitness, p_best_idx, g_fitness_real = self.compute_best_pop_params_with_diffusion_imgs_in_equal_state(
                    #         fitness_score, self.pops[i], success_indicator_index)
                    else:
                        if succeeded_pops.max() > self.p_best_n_success_imgs[i]:
                            self.update_iteration_params(fitness_score, i, self.pops[i], succeeded_pops,
                                                    success_indicator_index)


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
                        f"【{itr}/{self.max_iter}】Success: g_fitness: {self.g_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[success_indicator.cpu().data.numpy()][0]}, probability: {fitness_score[success_indicator.cpu().data.numpy()][0]}")
                    return True
            else:
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
                    Image.fromarray(current_adv_images_).save(fr'{self.save_dir}/{filename}_failed.png') # TODO: check is the save hear is needed and if the added 'failed' to the saved image name is needed as well

                if self.best_pop_on_src_and_diffusoin_count_imgs_success > 0:
                    print(
                        f"【{itr}/{self.max_iter}】Success on: {self.best_pop_on_src_and_diffusoin_count_imgs_success}, g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}")
                else:
                    print(f"【{itr}/{self.max_iter}】Failed: g_fitness: {self.g_best_fitness}, g_fitness_real: {self.sg_best_fitness}, p_fitness: {self.p_best_fitness[i]}, prediction: {pred_labels[0]}, probability: {fitness_score[0]}")

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
    def update_best_pop_attack(self, i, succeeded_pops):
        self.best_pop_on_src_and_diffusoin_count_imgs_success = succeeded_pops.max()
        best_pop = self.pops[i][succeeded_pops.argmax()]
        self.best_pop_on_src_and_diffusoin = best_pop
        # self.p_best[i] = best_pop
        self.g_best = self.p_best[i]
        self.sg_best = self.p_best[i]

    def run_pso(self, file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped, attack_with_diffusion: bool): #data_loader

        total = 0
        success_cnt = 0
        size = 224

        for i, (image, bbx, mask, filename) in enumerate(zip(cropped_resized_imgs, bbx, masks_cropped, file_names)): #data_loader # iterative each images
            self.mask = mask
            total += 1
            softmax_ori = self.calculate_fitness(image)
            fitness_score_ori, pred_label = torch.max(softmax_ori, dim=1)
            print(f"filename: {filename}.png predicted as: {pred_label.item()}")
            is_find_init = self.initialize(image, pred_label, filename)
            if is_find_init:
                success_cnt += 1
                print("Initial found!!!")
                print("==" * 30)
                continue

            for itr in range(self.max_iter):
                is_find_search = self.update(image, pred_label, itr, filename)
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
                                    attack_with_diffusion: bool): #data_loader

        total = 0
        success_cnt = 0
        size = 224

        for i, (image, bbx, mask, filename) in enumerate(zip(cropped_resized_imgs, bbx, masks_cropped, file_names)): #data_loader # iterative each images
            self.reset_attack_params()
            diffusion_imgs = DiffusionImages(filename, bbx, size, self.save_dir)
            self.mask = mask
            total += 1
            softmax_ori = self.calculate_fitness(image)
            fitness_score_ori, pred_label = torch.max(softmax_ori, dim=1)
            if attack_with_diffusion:
                is_find_init = self.initialize(image, pred_label, filename, diffusion_imgs)
            else:
                is_find_init = self.initialize(image, pred_label, filename, None)

            if is_find_init:
                success_cnt += 1
                print("Initial found!!!")
                print("==" * 30)
                self.save_adv_images(image, filename, attack_with_diffusion=attack_with_diffusion, diffusion_imgs=diffusion_imgs)
                continue

            for itr in range(self.max_iter):
                if attack_with_diffusion:
                    is_find_search = self.update(image, pred_label, itr, filename, diffusion_imgs)
                else:
                    is_find_search = self.update(image, pred_label, itr, filename, None)

                if is_find_search:
                    success_cnt += 1
                    print("==" * 30)
                    break

            self.save_adv_images(image, filename, attack_with_diffusion=attack_with_diffusion, diffusion_imgs=diffusion_imgs)

        asr = round(100 * (success_cnt / total), 2)

        return asr

    def save_adv_images(self, image: np.ndarray, filename: str, attack_with_diffusion: bool = False, diffusion_imgs: DiffusionImages = None):
        filename_without_ext = filename.split('.')[0]
        best_pop = self.best_pop_on_src_and_diffusoin
        if best_pop is None:
            best_pop = None #TODO: set best pop
        adv_img = self.gen_adv_images_by_pops(image, [best_pop])
        if attack_with_diffusion and not diffusion_imgs:
            raise Exception("attack_with_diffusion is True but diffusion_imgs is None ")

        image2saved_ = cv2.cvtColor(adv_img[0], cv2.COLOR_BGR2RGB)  # TODO: check is really needed
        if not diffusion_imgs:
            Image.fromarray(image2saved_).save(fr'{self.save_dir}/{filename}.png')
        else:
            if attack_with_diffusion:
                output_dir = diffusion_imgs.output_dir_special
            else:
                output_dir = diffusion_imgs.output_dir_normal
            print("output_dir: ", output_dir)
            os.makedirs(output_dir, exist_ok=True)
            Image.fromarray(image2saved_).save(f'{output_dir}/{filename}.png')
            Image.fromarray(image2saved_).save(f'{output_dir}/{filename}.png')

            for diffusion_img, diffusion_img_name in diffusion_imgs.diffusion_images_for_test():
                adv_img = self.gen_adv_images_by_pops(diffusion_img, [best_pop])
                image2saved_ = cv2.cvtColor(adv_img[0], cv2.COLOR_BGR2RGB)  # TODO: check is really needed
                Image.fromarray(image2saved_).save(fr'{output_dir}/{diffusion_img_name}.png')

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


if __name__ == "__main__":

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
                    save_dir=experiment_dir
                    )

    # # asr = pso.run_pso(file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped)

    asr_without_diffusion = pso.run_pso_with_diffusion_imgs(file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped, attack_with_diffusion=False)
    print(f"ASR without diffusion of {args.dataset_name}_{args.model_name} is: {asr_without_diffusion}")
    #
    asr_with_diffusion = pso.run_pso_with_diffusion_imgs(file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped, attack_with_diffusion=True)
    print(f"ASR with diffusion of {args.dataset_name}_{args.model_name} is: {asr_with_diffusion}")

    inference_on_src_attacked.main(args.model_wrapper[0].model_name, experiment_folder=experiment_dir, attack_methods=[ATTACK_TYPE_A, ATTACK_TYPE_B], save_results=True, is_adv_model= args.is_adv_model)

    if args.ensemble:
        # Currently we have at most 2 models in the ensemble
        inference_on_src_attacked.main(args.model_wrapper[1].model_name, experiment_folder=experiment_dir,
                                       attack_methods=[ATTACK_TYPE_A, ATTACK_TYPE_B], save_results=True,
                                       save_to_file_type='a', is_adv_model=not args.is_adv_model)

    if args.plot_pairs:
        create_pair_plots(experiment_dir)
    print("Finished !!!")
