import os
import cv2
import torch.cuda
from typing import List

import load_traffic_sign_models as traffic_model
from inferenceAndResults.results import Results


def inference_folder_with_attacked_images(attack_db: str, experiment_folder: str, save_to_file_type='w'):

    true_label = 12 if attack_db == 'LISA' else 14
    results = Results()
    model, pre_process = traffic_model.load_model(attack_db, attack_type='physical', target_model='normal')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for file_name in os.listdir(experiment_folder):
        if file_name.endswith("png"):
            pred_label = ingerence_helper(device, experiment_folder, file_name, model, pre_process)
            update_results_params(file_name, pred_label, results, true_label)

    results.save_and_display(experiment_folder, save_to_file_type)
def inference_folder_with_sub_attack_folders_with_attacked_images(attack_db: str, experiment_folder: str, attack_type: str=ATTACK_TYPE_A, save_to_file_type: str='w'):
    """

    Args:
        attack_db: 'LISA' or 'GTSRB'
        experiment_folder:
        attack_type:

    Returns:

    """
    true_label = 12 if attack_db == 'LISA' else 14
    results = Results()
    model, pre_process = traffic_model.load_model(attack_db, attack_type='physical', target_model='normal')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for file_dir_name in os.listdir(experiment_folder):
        if file_dir_name != 'black_box':
            file_dir = os.path.join(experiment_folder, file_dir_name)
            if os.path.isdir(file_dir):
                images_folder = os.path.join(file_dir, attack_type)
                for file_name in os.listdir(images_folder):
                    if file_name.endswith("png"):
                        results.total_images += 1
                        pred_label = ingerence_helper(device, images_folder, file_name, model, pre_process)
                        update_results_params(file_name, pred_label, results, true_label)

    results.save_and_display(experiment_folder, save_to_file_type=save_to_file_type)


def update_results_params(file_name: str, pred_label: int, results: Results, true_label: int):
    if 'adv_img' in file_name or 'road_' in file_name:
        results.total_src_images += 1
        if true_label == pred_label:
            results.model_pred_correctly_on_srcAdv_img += 1
    else:
        results.total_diffusion_images += 1
        if true_label == pred_label.item():
            results.total_diffusion_images_model_pred_correctly += 1
        else:
            results.total_diffusion_imgs_attacked += 1

def ingerence_helper(device: str, experiment_folder: str, file_name: str, model: torch.nn.Module, pre_process: callable) -> int:
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
    image = cv2.resize(image, (32, 32))
    image = pre_process(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        predict = torch.softmax(model(image), 1)
        predict = torch.mean(predict, dim=0)
        pred_label = torch.argmax(predict)

        return pred_label
    #         if pred_label != true_label:
    #             total_img_attacked += 1
    # return total_img_attacked, total_imgs

def main(attack_db: str, experiment_folder: str, attack_methods: List[str] = [ATTACK_TYPE_A, ATTACK_TYPE_B]):
    print(f"attack_db: {attack_db}")
    print("experiment_folder: ", experiment_folder)
    save_to_file_type = 'w'

    for i, attack_method in enumerate(attack_methods):
        if i > 0:
            save_to_file_type = 'a'
        print(f"attack_method: {attack_method}")
        inference_folder_with_sub_attack_folders_with_attacked_images(attack_db, experiment_folder, attack_method, save_to_file_type)
        print('#' * 100)

if __name__ == "__main__":
    attack_db = 'LISA'
    experiment_folder = r'/tmp/pycharm_project_250/RFLA/larger_images_experiments/physical_attack_RFLA_LISA_shape-hexagon_maxIter-200'
    main(attack_db, experiment_folder, [ATTACK_TYPE_A, ATTACK_TYPE_B]) #, ['normal_atatck', 'special_atatck']
