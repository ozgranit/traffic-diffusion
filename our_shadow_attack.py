import argparse
import sys
from typing import Union, Tuple
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from attacks.ShadowAttack.pso import PSO
from attacks.ShadowAttack.utils import *
from attacks.ShadowAttack.our_pso import OurPSO
from buildData.diffusion_generated_imgs import DiffusionImages
from buildData.input_data import InputData
from datasets.kaggle.kaggle_images_settings import KAGGLE_IMAGES
from datasets.larger_images.larger_images_settings import LARGER_IMAGES
from general_utils import str2bool
from inferenceAndResults import inference_on_src_attacked
from models.gtsrb_model import GtsrbModel
from models.lisa_model import LisaModel
from settings import ATTACK_TYPE_A, ATTACK_TYPE_B, LISA, GTSRB, \
                     STOP_SIGN_LISA_LABEL, STOP_SIGN_GTSRB_LABEL
from load_images import process_image
from plot_images import create_pair_plots
from shadow import Shadow

sys.path.append('attacks/ShadowAttack')
from attacks.ShadowAttack.utils import brightness, judge_mask_type, load_mask
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Set this to False for fully deterministic results
torch.use_deterministic_algorithms(True)

class Attack:

    def __init__(self, args: argparse.ArgumentParser):
        self.shadow_level: float = args.shadow_level
        self.target_model: str = args.target_model
        self.is_adv_model: bool = False if self.target_model == 'normal' else True
        self.attack_model_name: str = args.attack_model_name
        self.attack_type: str = args.attack_type
        self.input_data: Union[str, InputData] = args.input_data
        self.mask_path: str = args.mask_path
        self.image_label: int = args.image_label
        self.polygon: int = args.polygon
        self.n_try: int = args.n_try
        self.iter_num: int = args.iter_num
        self.output_dir: str = args.output_dir
        self.our_attack: bool = args.our_attack
        self.with_EOT: bool = args.with_EOT
        self.untargeted_only: bool = args.untargeted_only
        self.ensemble: bool = args.ensemble
        self.transform_num = args.transform_num

        self.validate_input_params()
        self.set_params_by_attack_type()

        self.set_model_wrapper(args)

    def set_model_wrapper(self, args: argparse.ArgumentParser):
        if self.attack_model_name == LISA:
            if self.image_label is None or self.image_label == 'None':
                self.image_label = STOP_SIGN_LISA_LABEL
            self.model_wrapper = LisaModel(self.is_adv_model, args.crop_size)
            if self.ensemble:
                self.model_wrapper = [self.model_wrapper, LisaModel(not self.is_adv_model, args.crop_size)]
            else:
                self.model_wrapper = [self.model_wrapper]

        else:
            if self.image_label is None or self.image_label == 'None':
                self.image_label = STOP_SIGN_GTSRB_LABEL
            self.model_wrapper = GtsrbModel(self.is_adv_model, args.crop_size)
            if self.ensemble:
                self.model_wrapper = [self.model_wrapper, GtsrbModel(not self.is_adv_model, args.crop_size)]
            else:
                self.model_wrapper = [self.model_wrapper]

    def set_params_by_attack_type(self):
        if self.attack_type == 'digital':
            self.particle_size = 10
            # iter_num = 100
            self.x_min, self.x_max = -16, 48
            self.max_speed = 1.5
        else:
            self.particle_size = 10
            # self.iter_num = 200
            self.x_min, self.x_max = -112, 336
            self.max_speed = 10.
            # self.n_try = 1

    def validate_input_params(self):
        if not self.with_EOT:
            assert self.transform_num == 0

        if self.our_attack:
            assert isinstance(self.input_data, InputData)
        else:
            assert os.path.isfile(self.input_data)

        assert self.attack_type in ['digital', 'physical']

    def attack(self, attack_image: np.ndarray, label: int, coords: Tuple[np.ndarray, np.ndarray],
               targeted_attack: bool = False,
               physical_attack: bool=False, **parameters):
        r"""
        Physical-world adversarial attack by shadow.

        Args:
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

        for attempt in range(self.n_try):

            if succeed:
                break

            print(f"try {attempt + 1}:", end=" ")
            if self.our_attack:
                pso = OurPSO(self.with_EOT, self.polygon * 2, self.particle_size, self.iter_num, self.x_min, self.x_max,
                          self.max_speed, self.shadow_level, attack_image, coords,
                          self.model_wrapper, targeted_attack, physical_attack, label, self.model_wrapper[0].pre_process_image, **parameters)

            else:
                pso = PSO(self.with_EOT, self.polygon * 2, self.particle_size, self.iter_num, self.x_min, self.x_max,
                          self.max_speed, self.shadow_level, attack_image, coords,
                          self.model_wrapper[0].model, targeted_attack, physical_attack, label, self.model_wrapper[0].pre_process_image, **parameters)

            best_solution, best_pos, succeed, query = pso.update_digital() \
                if not physical_attack else pso.update_physical()

            if targeted_attack:
                best_solution = 1 - best_solution
            print(f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}")
            if best_solution < global_best_solution:
                global_best_solution = best_solution
                global_best_position = best_pos
            num_query += query

        coefficient = 3
        adv_image, shadow_area = draw_shadow(
            global_best_position, attack_image, coords, self.shadow_level)
        adv_image = shadow_edge_blur(adv_image, shadow_area, coefficient)


        return adv_image, succeed, num_query, Shadow(global_best_position, coords, self.shadow_level, coefficient)

    def attack_digital(self):
        save_dir = f'./adv_img/{self.attack_model_name}/{int(self.shadow_level * 100)}'
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            for name in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, name))

        with open(f'./dataset/{self.attack_model_name}/test.pkl', 'rb') as dataset:
            test_data = pickle.load(dataset)
            images, labels = test_data['data'], test_data['labels']

        for index in range(len(images)):
            mask_type = judge_mask_type(self.attack_model_name, labels[index])
            if brightness(images[index], self.mask_list[mask_type]) >= 120:
                adv_img, success, num_query = attack(
                    images[index], labels[index], position_list[mask_type])
                cv2.imwrite(f"{save_dir}/{index}_{labels[index]}_{num_query}_{success}.bmp", adv_img)

        print("Attack finished! Success rate: ", end='')
        print(Counter(map(lambda x: x[:-4].split('_')[-1],
                          os.listdir(save_dir)))['True'] / len(os.listdir(save_dir)))

    def attack_physical(self):
        global position_list

        mask_image = cv2.resize(
            cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED), (224, 224))
        target_image = cv2.resize(
            cv2.imread(self.input_data), (224, 224))
        pos_list = np.where(mask_image.sum(axis=2) > 0)

        # EOT is included in the first stage
        adv_img, succeed, num_query, shadow_best_params = self.attack(target_image, self.image_label, pos_list,
                               physical_attack=True, transform_num=self.transform_num)

        confidence, pred_label, msg, summary_msg, failed = self.check_final_attacked_image(adv_img, save_image=True)

        # Predict stabilization
        adv_img, succeed, num_query, shadow_best_params = self.attack(target_image, self.image_label, pos_list, targeted_attack=True,
                               physical_attack=True, target=pred_label, transform_num=self.transform_num)
        confidence, pred_label, msg, summary_msg, failed = self.check_final_attacked_image(adv_img, save_image=True)
        if not failed:
            print(summary_msg)

        adv_img_to_plot = cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB)
        plt.imshow(adv_img_to_plot)
        plt.show()
        # cv2.imshow("Adversarial image", adv_img)
        # cv2.waitKey(0)

    def our_attack_physical_wrapper(self, attack_with_diffusion: bool = False) -> float:
        size = 224
        file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, masks_cropped = process_image(
            self.input_data.input_folder,  # TODO: remove comment and add notes that kaggle can be loaded, #kaggle_images',
            self.input_data.annotations_folder,
            self.attack_model_name, crop_size=size, mask_folder=self.input_data.mask_folder)

        total_orig_success = 0
        total_orig_images = len(file_names)
        self.physical_experiment_dir = f'{self.output_dir}/physical_attack_untargeted-{int(self.untargeted_only)}_{self.attack_model_name}_EOT-{int(self.with_EOT)}_ensemble-{int(self.ensemble)}_shadowLevel-{self.shadow_level}_iter-{self.iter_num}'
        fixed_mask = None
        if input_data.mask_folder is None:
            mask_path = r'octagon_mask.png'
            fixed_mask = cv2.resize(
                cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (size, size))

        for ind in range(len(file_names)):
            diffusion_imgs = DiffusionImages(file_names[ind], bbx[ind], size, self.physical_experiment_dir)
            confidence, pred_label, success, msg = self.model_wrapper[0].test_single_image(cropped_resized_imgs[ind], labels[ind], print_results=False)
            print(f"Prediction on clean orig image {file_names[ind]} is: {pred_label} with conf: {confidence}")
            transform_num = self.transform_num
            if attack_with_diffusion:
                if self.with_EOT:
                    # Setting transform_num for normal attack (method A) to be equivalent to total transformed created images in special attack (method B)
                    transform_num = self.transform_num * len(diffusion_imgs.generated_imgs_train_cropped_names) + self.transform_num
                mask = fixed_mask if input_data.mask_folder is None else masks_cropped[ind]
                success, adv_img, best_shadow_params = self.our_attack_physical(cropped_resized_imgs[ind], file_names[ind],
                                         mask, labels[ind], transform_num=transform_num,
                                                   diffusion_imgs=diffusion_imgs)
            else:
                success, adv_img, best_shadow_params = self.our_attack_physical(cropped_resized_imgs[ind], file_names[ind],
                                         masks_cropped[ind], labels[ind], transform_num=transform_num)

            if success:
                total_orig_success += 1

            self.save_adv_images(adv_img,
                                 cropped_resized_imgs[ind],
                                 file_names[ind],
                                 best_shadow_params=best_shadow_params,
                                 attack_with_diffusion=attack_with_diffusion,
                                 diffusion_imgs=diffusion_imgs,
                                 experiment_dir=self.physical_experiment_dir)

        asr_orig = round(100 * (total_orig_success / total_orig_images), 2)

        return asr_orig

    def our_attack_physical(self, image: np.ndarray, image_name: str,
                                         mask: np.ndarray, true_label: int,
                                        transform_num: int = 0,
                                                   diffusion_imgs: DiffusionImages = None) -> bool:
        mask_image = np.expand_dims(mask, axis=-1)
        pos_list = np.where(mask_image.sum(axis=2) > 0)

        # EOT might be included in the first stage - only is self.with_EOT is True
        generated_images = None if diffusion_imgs is None else diffusion_imgs.generated_images_for_special_attack
        adv_img, succeed, num_query, shadow_best_params = self.attack(image,
                                                                        true_label,
                                                                        coords=pos_list,
                                                                        targeted_attack=False,
                                                                        physical_attack=True,
                                                                        transform_num=transform_num,
                                                                        generated_images=generated_images)  # generated_imgs_cropped

        confidence, pred_label, msg, summary_msg, failed = self.check_final_attacked_image(adv_img, save_image = False)

        if not self.untargeted_only:
            adv_img, succeed, num_query, shadow_best_params = self.attack(image,
                                                                            true_label,
                                                                            pos_list,
                                                                            targeted_attack=True,
                                                                            physical_attack=True,
                                                                            target = pred_label,
                                                                            transform_num=transform_num,
                                                                            generated_images=generated_images)  # generated_imgs_cropped
            # Predict stabilization

            confidence, pred_label, msg, summary_msg, failed = self.check_final_attacked_image(adv_img, save_image=False)
            if not failed:
                print(summary_msg)

        return not failed, adv_img, shadow_best_params

    def check_final_attacked_image(self, adv_img: np.ndarray, save_image: bool = True):
        if save_image:
            adv_img_path = f'./{self.output_dir}/adv_img.png'
            os.makedirs(os.path.split(adv_img_path)[0], exist_ok=True)
            cv2.imwrite(adv_img_path, adv_img)
            confidence, pred_label, failed, msg = self.model_wrapper[0].test_single_image(adv_img_path,
                                                                        self.image_label,
                                                                        self.is_adv_model)
        else:
            confidence, pred_label, failed, msg = self.model_wrapper[0].test_single_image(adv_img,
                                                                        self.image_label,
                                                                        self.is_adv_model)

        if failed:
            summary_msg = 'Attack failed! Try to run again.'
            print(summary_msg)
        else:
            summary_msg = 'Attack succeed! Try to implement it in the real world.'

        return confidence, pred_label, msg, summary_msg, failed

    def save_adv_images(self, adv_image, clean_image, image_name: str, best_shadow_params: Shadow,
                        attack_with_diffusion: bool, diffusion_imgs: DiffusionImages, experiment_dir: str):
        if attack_with_diffusion and not diffusion_imgs:
            raise Exception("attack_with_diffusion is True but diffusion_imgs is None ")
        # image2saved = cv2.cvtColor(adv_image, cv2.COLOR_BGR2RGB)  # TODO: check is really needed
        if not diffusion_imgs:
            cv2.imwrite(f"{experiment_dir}/{image_name}.png", adv_image)
        else:
            if attack_with_diffusion:
                output_dir = diffusion_imgs.output_dir_special
            else:
                output_dir = diffusion_imgs.output_dir_normal
        #     print("output_dir: ", output_dir)
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{image_name}.png", adv_image)

            for diffusion_img, diffusion_img_name in diffusion_imgs.diffusion_images_for_test():
                diffusion_adv_img = self.add_shadow_attack_to_image(best_shadow_params, diffusion_img)
                diffusion_adv_img = diffusion_adv_img[0] if len(diffusion_adv_img.shape) > 3 else diffusion_adv_img
                # diffusion_adv_img = cv2.cvtColor(diffusion_adv_img, cv2.COLOR_BGR2RGB)  # TODO: check is really needed
                cv2.imwrite(f"{output_dir}/{diffusion_img_name}.png", diffusion_adv_img)

    def add_shadow_attack_to_image(self, shadow_params_normal: Shadow, test_image):
        test_image_shadow, shadow_area = draw_shadow(
            shadow_params_normal.global_best_position, test_image, shadow_params_normal.coords,
            shadow_params_normal.shadow_level)
        coefficient = 3
        test_image_shadow = shadow_edge_blur(test_image_shadow, shadow_area, coefficient)
        return test_image_shadow

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adversarial attack by shadow")
    parser.add_argument("--shadow_level", type=float, default=0.43,
                        help="Shadow coefficient k")
    parser.add_argument("--input_data_name", type=str, default=LARGER_IMAGES,
                        help=f"If you use InputData then you can set {LARGER_IMAGES} or '{KAGGLE_IMAGES}'")
    parser.add_argument("--attack_model_name", type=str, default=GTSRB,
                        help="The target dataset should be specified for a digital attack")
    parser.add_argument("--attack_type", type=str, default="physical",
                        help="Digital attack or physical attack")
    parser.add_argument("--input_data", type=str, default="attacks/ShadowAttack/tmp/gtsrb_30.png",
                        #"ShadowAttack/tmp/gtsrb_30.png",
                        help=f"A file path OR input_data class to the target image/folder should be specified for a physical attack. "
                             "WARNING: If arg `input_data_name` is not None then this param will be ignored")
    parser.add_argument("--mask_path", type=str, default="attacks/ShadowAttack/tmp/gtsrb_30_mask.png",
                        help="A file path to the mask should be specified for a physical attack")
    parser.add_argument("--image_label", default=None,
                        help="Type: int."
                             "A ground truth should be specified for a physical attack, if None, takes label for stop sign depends on model Lisa or Gtsrb")
    parser.add_argument("--polygon", type=int, default=3,
                        help="Shadow shape: n-sided polygon")
    parser.add_argument("--n_try", type=int, default=1,
                        help="n-random-start strategy: retry n times, usually for digital-5 and for physical 1")
    parser.add_argument("--target_model", type=str, default="normal",
                        help="Attack normal model or robust model")
    parser.add_argument("--iter_num", type=int, default=200,
                        help="Usually 100 for normal attack and 200 for physical attack")
    parser.add_argument("--crop_size", type=int, default=32,
                        help="Image size before feed in to the model")
    parser.add_argument("--output_dir", type=str, default="experiments/_tmp_/shadowAttack", #rf'experiments/{input_data.input_name}',
                        help="Folder path to dave output")
    parser.add_argument("--our_attack", type=str2bool, default=True,
                        # action="store_true",
                        help="Should apply our attack, if True please put a folder in image_path otherwise a path for an image")
    parser.add_argument("--with_EOT", type=str2bool, default=False,
                        # action="store_true",
                        help="Weather to apply EOT transformations")
    parser.add_argument("--untargeted_only", type=str2bool, default=False,
                        # action="store_true",
                        help="Weather to apply stabilization attack")
    parser.add_argument("--ensemble", type=str2bool, default=False,
                        # action="store_true",
                        help="If True then the attack is applied against both regular and robust(model_adv) models")
    parser.add_argument("--transform_num", type=int, default=0,
                        help="Number of EOT transformations. "
                             "If our physical attack is applied then transform_num for normal attack will be expanded to match number of total transformed images")
    parser.add_argument("--plot_pairs", type=str2bool, default= False,
                        # action="store_true",
                        help="If True then the attack is applied against both regular and robust(model_adv) models")

    args = parser.parse_args()
    if args.input_data_name is not None and args.input_data_name != "None":
        input_data = InputData(args.input_data_name)
        args.input_data = input_data
        args.output_dir = os.path.join(args.output_dir, input_data.input_name.lower())
    attack = Attack(args)

    if args.attack_type == 'digital':
        attack.attack_digital()
    else:   # Physical attack
        if not args.our_attack:
            attack.attack_physical()
        else:
            asr_orig_without_diffusion = attack.our_attack_physical_wrapper(attack_with_diffusion= False)
            print(f"ASR orig without diffusion is: {asr_orig_without_diffusion}")

            asr_orig_with_diffusion = attack.our_attack_physical_wrapper(attack_with_diffusion= True)
            print(f"ASR orig with diffusion is: {asr_orig_with_diffusion}")

            inference_on_src_attacked.main(attack.model_wrapper[0].model_name, experiment_folder=attack.physical_experiment_dir,
                                           attack_methods=[ATTACK_TYPE_A, ATTACK_TYPE_B], save_results=True)
            if args.plot_pairs:
                create_pair_plots(attack.physical_experiment_dir)
            print("Finished !!!")



