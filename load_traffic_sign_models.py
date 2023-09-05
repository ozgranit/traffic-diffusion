# def load_params():
import json
import torch
from torchvision import transforms
from attacks.ShadowAttack import gtsrb, lisa
from attacks.ShadowAttack.shadow_attack_settings import PARAMS_PATH, MODEL_PATH
from attacks.ShadowAttack.utils import pre_process_image
from settings import GTSRB, LISA


def load_model(attack_db = GTSRB, attack_type = 'physical', target_model = 'normal'):
    with open(PARAMS_PATH, 'rb') as f:
        params = json.load(f)
        class_n_gtsrb = params[GTSRB]['class_n']
        class_n_lisa = params[LISA]['class_n']
        device = params['device']

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
    assert attack_db in [LISA, GTSRB]
    if attack_db == LISA:
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