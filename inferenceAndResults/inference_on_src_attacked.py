import os
import cv2
import torch.cuda
from typing import List, Tuple

from inferenceAndResults.results import Results
from models.gtsrb_model import GtsrbModel
from models.lisa_model import LisaModel
from settings import ATTACK_TYPE_A, ATTACK_TYPE_B, STOP_SIGN_LISA_LABEL, LISA, STOP_SIGN_GTSRB_LABEL, DEVICE, GTSRB

def inference_folder_with_attacked_images(model_name: str, experiment_folder: str,
                                          save_results: bool = True, save_to_file_type='w',
                                          is_adv_model: bool = False, crop_size: int = 32):

    # true_label = 12 if attack_db == 'LISA' else 14
    # model, pre_process = traffic_model.load_model(attack_db, attack_type='physical', target_model='normal')
    model_wrapper, true_label = load_model_and_set_true_label(is_adv_model, model_name, crop_size)


    results = Results(model_name, experiment_folder, is_adv_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for file_name in os.listdir(experiment_folder):
        if file_name.endswith("png") and not file_name.endswith("_mask.png"):
            pred_label = inference_helper(device, experiment_folder, file_name, model_wrapper.model, model_wrapper.pre_process_image, crop_size=crop_size) # TODO: remove this
            image_path = os.path.join(experiment_folder, file_name)
            confidence, pred_label, model_failed, msg = model_wrapper.test_single_image(image_path, true_label,
                                                                                        print_results=False)

            update_results_params(file_name, pred_label, results, true_label)

    results.save_and_display(experiment_folder, save_results, save_to_file_type)
def inference_folder_with_sub_attack_folders_with_attacked_images(model_name: str, experiment_folder: str, attack_type: str = ATTACK_TYPE_A,
                                                                  save_results: bool = True, save_to_file_type: str='w',
                                                                  is_adv_model: bool = False, crop_size: int = 32):
    """

    Args:
        model_name: 'LISA' or 'GTSRB'
        experiment_folder: path to experiment_folder
        attack_type: METHOD_A or METHOD_B
        save_results: if True then results are saved to file in the experiment_folder
        save_to_file_type: if 'w' then starts a new document, if 'a' append to existing document if exists otherwise starts a new one
        is_adv_model: if True then adv ('robust') model is used, if False then 'normal' model is used
        crop_size: size of image before feeding to the model
    Returns:

    """

    results = Results(model_name, experiment_folder, is_adv_model)
    model_wrapper, true_label = load_model_and_set_true_label(is_adv_model, model_name, crop_size)
    device = DEVICE
    for file_dir_name in os.listdir(experiment_folder):
        if file_dir_name.lower() != 'black_box':
            success_at_least_one_diffusion_image_of_image_dir = False
            file_dir = os.path.join(experiment_folder, file_dir_name)
            if os.path.isdir(file_dir):
                images_folder = os.path.join(file_dir, attack_type)
                for file_name in os.listdir(images_folder):
                    if file_name.endswith("png") and not file_name.endswith("_mask.png"):
                        results.total_images += 1
                        # pred_label_ = inference_helper(device, images_folder, file_name, model_wrapper.model, model_wrapper.pre_process_image)  #TODO: remove this line
                        image_path = os.path.join(images_folder, file_name)
                        confidence, pred_label, attack_failed, msg = model_wrapper.test_single_image(image_path, true_label, print_results=False)
                        update_results_params(file_name, pred_label, results, true_label)
                        if not attack_failed and not success_at_least_one_diffusion_image_of_image_dir and 'adv_img' not in file_name and 'road_' not in file_name:
                            success_at_least_one_diffusion_image_of_image_dir = True
                            results.total_diff_imgs_with_at_lease_one_diffusion_image_success += 1

    results.save_and_display(experiment_folder, save_results, save_to_file_type=save_to_file_type, is_adv_model= is_adv_model)


def load_model_and_set_true_label(adv_model: bool, attack_db: str, crop_size: int):
    if attack_db == LISA:
        true_label = STOP_SIGN_LISA_LABEL
        model_wrapper = LisaModel(adv_model=adv_model, crop_size=crop_size)
    elif attack_db == GTSRB:
        true_label = STOP_SIGN_GTSRB_LABEL
        model_wrapper = GtsrbModel(adv_model=adv_model, crop_size=crop_size)
    else:
        raise Exception(f"attack_db has to be {LISA} or {GTSRB}")
    return model_wrapper, true_label


def update_results_params(file_name: str, pred_label: int, results: Results, true_label: int):
    if 'adv_img' in file_name or 'road_' in file_name:
        results.total_src_images += 1
        if true_label == pred_label:
            results.model_pred_correctly_on_srcAdv_img += 1
    else:
        results.total_diffusion_images += 1
        if isinstance(pred_label, torch.Tensor):
            pred_label = pred_label.item()
        if true_label == pred_label:
            results.total_diffusion_images_model_pred_correctly += 1
        else:
            results.total_diffusion_imgs_attacked += 1

def inference_helper(device: str, experiment_folder: str, file_name: str,
                     model: torch.nn.Module, pre_process: callable, crop_size: int = 32) -> int:
    """
       Perform inference on an image using a PyTorch model.

       Args:
           device (str): The device to use for inference, e.g., 'cuda' or 'cpu'.
           experiment_folder (str): The folder containing the image file.
           file_name (str): The name of the image file.
           model (torch.nn.Module): The PyTorch model for inference.
           pre_process (callable): A function to preprocess the input image.

       Returns:
           int: The predicted label for the input image.
    """

    image_path = os.path.join(experiment_folder, file_name)
    image = cv2.imread(image_path)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    image = cv2.resize(image, crop_size)
    image = pre_process(image)
    if len(image.shape)<=3:
        image = image.unsqueeze(0).to(device)
    image = image.to(device)
    with torch.no_grad():
        predict = torch.softmax(model(image), 1)
        predict = torch.mean(predict, dim=0)
        pred_label = torch.argmax(predict)

        return pred_label.item()
    #         if pred_label != true_label:
    #             total_img_attacked += 1
    # return total_img_attacked, total_imgs

def main(model_name: str, experiment_folder: str, attack_methods: List[str] = [ATTACK_TYPE_A, ATTACK_TYPE_B],
         save_results: bool = True, save_to_file_type: str = 'w', is_adv_model: bool = False):
    print(f"Model name: {model_name}, Is adv (robust) model: {is_adv_model}")
    print("Experiment_folder: ", experiment_folder)

    for i, attack_method in enumerate(attack_methods):
        if i > 0:
            save_to_file_type = 'a'
        print(f"attack_method: {attack_method}")
        inference_folder_with_sub_attack_folders_with_attacked_images(model_name, experiment_folder, attack_method, save_results, save_to_file_type, is_adv_model)
        print('#' * 100)

if __name__ == "__main__":
    model_name = GTSRB
    adv_model = False
    experiment_folder = r'experiments/RFLA/larger_images/physical_attack_RFLA_GTSRB_isAdv-1_shape-hexagon_maxIter-200_ensemble-0_interploate-0'
    main(model_name, experiment_folder, [ATTACK_TYPE_A, ATTACK_TYPE_B], save_results=False, save_to_file_type='w', is_adv_model=adv_model) #, ['normal_atatck', 'special_atatck']
