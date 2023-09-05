from typing import Tuple

from models.gtsrb_model import GtsrbModel
from models.lisa_model import LisaModel
from models.base_model import BaseModel
from settings import LISA, GTSRB


def load_model_wrapper(model_name: str, adv_model: bool = False, crop_size: Tuple[int, int] = (32, 32)) -> BaseModel:
    if model_name == LISA:
        model_wrapper = LisaModel(adv_model, crop_size)
    elif model_name == GTSRB:
        model_wrapper = GtsrbModel(adv_model, crop_size)

    return model_wrapper


