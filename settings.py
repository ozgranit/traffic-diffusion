import torch.cuda

GENERATED_IMAGES_TYPES_TRAIN = ['midday', 'cloud', 'rain']
GENERATED_IMAGES_TYPES_TEST = ['night', 'snow', 'dawn']
GENERATED_IMAGES_TYPES_ALL = GENERATED_IMAGES_TYPES_TRAIN + GENERATED_IMAGES_TYPES_TEST

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LISA = 'LISA'
GTSRB = 'GTSRB'

STOP_SIGN_LISA_LABEL = 12
STOP_SIGN_GTSRB_LABEL = 14

ATTACK_TYPE_A = 'normal_attack'
ATTACK_TYPE_B = 'special_attack'
