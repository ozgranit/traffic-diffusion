import os
import torch
import cv2
import pandas as pd
import numpy as np
from attacks.ShadowAttack import gtsrb, lisa
from attacks.ShadowAttack.shadow_attack_settings import PARAMS_PATH, MODEL_PATH
from attacks.ShadowAttack.utils import pre_process_image
from torchvision import transforms
import json

from settings import GTSRB, LISA, ATTACK_TYPE_A, ATTACK_TYPE_B

with open(PARAMS_PATH, 'rb') as f:
    params = json.load(f)
    class_n_gtsrb = params[GTSRB]['class_n']
    class_n_lisa = params[LISA]['class_n']
    device = params['device']
    # position_list, mask_list = load_mask()

def predict_image(image: np.ndarray, model: torch.nn, description="", print_results=True, model_name='LISA'):
    with torch.no_grad():
        img_ = cv2.resize(image, (32, 32))
        if model_name == GTSRB:
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

# def perform_inference(image_path: str, model: torch.nn) -> bool:
#     # Load and preprocess the image
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (48, 48))  # Resize to the model's input size
#     img = img / 255.0  # Normalize pixel values
#
#     # Perform inference
#     prediction = model.predict(np.expand_dims(img, axis=0))
#     predicted_class = np.argmax(prediction)
#
#     # Return True if the predicted class is 14 (or False otherwise)
#     return predicted_class == 14


def load_model(model_name: str) -> [torch.nn, callable]:
    target_model = 'normal'
    assert model_name in [LISA, GTSRB]
    if model_name == LISA:
        model = lisa.LisaCNN(n_class=class_n_lisa).to(device)
        model.load_state_dict(
            # torch.load(f'{MODEL_PATH}/{"adv_" if target_model == "robust" else ""}model_lisa.pth',
            torch.load(f'{MODEL_PATH}/{"adv_" if target_model == "robust" else ""}model_lisa.pth',
                       map_location=torch.device(device)))
        pre_process = transforms.Compose([transforms.ToTensor()])
    else:
        model = gtsrb.GtsrbCNN(n_class=class_n_gtsrb).to(device)
        model.load_state_dict(
            # torch.load(f'{MODEL_PATH}/{"adv_" if target_model == "robust" else ""}model_gtsrb.pth',
            torch.load(f'{MODEL_PATH}/{"adv_" if target_model == "robust" else ""}model_gtsrb.pth',
                       map_location=torch.device(device)))
        pre_process = transforms.Compose([
            pre_process_image, transforms.ToTensor()])
    model.eval()

    return model, pre_process


def inference_on_experiment(base_dir, model_name):
    """
    for black-box inference, choose different model from the model used in the attacl
    Args:
        base_dir:

    Returns:

    """
    # Define the directory containing your images

    # Create a DataFrame to store results
    total_imgs = 0
    total_src_imgs = 0
    total_imgs_attacked = 0
    total_gen_imgs = 0
    total_gen_imgs_attacked = 0

    total_gen_normal_attacked = 0
    total_src_imgs_normal_attacked = 0

    total_gen_special_attacked = 0
    total_src_imgs_special_attacked = 0

    SRC_IMAGE = "adv_image" # Not generated image

    results_df = pd.DataFrame()
    ORIG_LABEL = 14 if model_name=='GTSRB' else 12
    model = load_model(model_name)
    # Iterate through subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            # Initialize a dictionary to store results for this road
            road_results = {"road": subdir}

            # Iterate through image folders ATTACK_TYPE_A and ATTACK_TYPE_B
            for attack_type in [ATTACK_TYPE_A, ATTACK_TYPE_B]:
                attack_path = os.path.join(subdir_path, attack_type)
                if os.path.exists(attack_path):
                    # Iterate through the images in the list
                    for image_name in os.listdir(attack_path):
                        img_path = os.path.join(attack_path, image_name)

                        # Check if the image exists and doesn't contain 'cmp' in its name
                        if os.path.exists(img_path) and 'cmp' not in img_path and img_path.endswith('png'):
                            total_imgs+=1
                            # Determine the image type based on the name
                            image_type = image_name.split('adv')[0]
                            if len(image_type) == 0:
                                total_src_imgs += 1
                                image_type = SRC_IMAGE
                            else:
                                total_gen_imgs+=1
                            img = cv2.imread(img_path)
                            # Perform inference and store the result in the dictionary
                            prob, label = predict_image(img, model, description=attack_type, print_results=False, model_name=model_name)
                            probability = prob.item()
                            label = label.item()
                            column_name_label = f"{attack_type}_{image_type}_Label"
                            column_name_prob = f"{attack_type}_{image_type}_Probability"
                            column_name_attack_succeded = f"{attack_type}_{image_type}_Probability"
                            road_results[column_name_label] = label
                            road_results[column_name_prob] = probability
                            road_results[column_name_prob] = 1 if column_name_label != ORIG_LABEL else 0
                            road_results[f"{column_name_label}_Type"] = attack_type  # Add attack type info
                            if label != ORIG_LABEL:
                                total_imgs_attacked+=1
                                if image_type != SRC_IMAGE:
                                    total_gen_imgs_attacked+=1
                                    if 'normal' in attack_type:
                                        total_gen_normal_attacked += 1
                                    else:
                                        total_gen_special_attacked += 1

                                else:
                                    if 'normal' in attack_type:
                                        total_src_imgs_normal_attacked += 1
                                    else:
                                        total_src_imgs_special_attacked += 1


            # Append the results for this road to the DataFrame
            results_df = results_df.append(road_results, ignore_index=True)

    # Save the results to a CSV file
    output_dir = os.path.join(base_dir, 'black_box')
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, f'{model_name}_result.csv'), index=False)

    print("total_imgs:", total_imgs)
    print("total_src_imgs: ", total_src_imgs)
    print("total_imgs_attacked:", total_imgs_attacked)
    print("total_gen_imgs:", total_gen_imgs)
    print("total_gen_imgs_attacked:", total_gen_imgs_attacked)

    print('-'*15)

    print("total_gen_normal_attacked:", total_gen_normal_attacked)
    print("total_src_imgs_normal_attacked:", total_src_imgs_normal_attacked)
    print("total_gen_special_attacked:", total_gen_special_attacked)
    print("total_src_imgs_special_attacked:", total_src_imgs_special_attacked)


if __name__ == "__main__":
    # # Checking GTSRB attack on LISA
    # model_name = 'LISA'
    # experiment_dir = r'larger_images/physical_attack_untar_mask_equal_split_GTSRB_EOT-False_iter-300'

    # # Checking LISA attack on GTSRB
    model_name = GTSRB  #| LISA
    experiment_dir = r'/tmp/pycharm_project_250/larger_images/experiments/physical_attack_untargeted_mask_equal_split_LISA_EOT-False_iter-200_level-0.43'

    attack_method = ATTACK_TYPE_A     #"normal_atatck
    inference_on_experiment(experiment_dir, model_name, attack_method)

    attack_method = ATTACK_TYPE_B     #"special_atatck
    inference_on_experiment(experiment_dir, model_name, attack_method)

