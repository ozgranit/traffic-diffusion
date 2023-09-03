import json
import os
from abc import abstractclassmethod
from typing import Union

import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch

from ShadowAttack.config import PARAMS_PATH

class ModelBase(ABC):

    def load_params(self, model_name: str):
        """

        Args:
            model_name: can be 'LISA' or 'GTSRB'

            loads self.params, self.class_n
        """
        with open(PARAMS_PATH, 'r') as config:
            self.params = json.load(config)
            self.class_n = self.params[model_name.upper()]['class_n']
            # self.position_list, _ = load_mask()

    def predict_img(self, img):
        predict = torch.softmax(self.model(img)[0], 0)
        pred_label = int(torch.argmax(predict).data)
        confidence = float(predict[pred_label].data)

        return confidence, pred_label

    def load_img_if_needed(self, img: Union[str, np.ndarray]):
        # If img is path
        if isinstance(str, img):
            if os.path.exists(img):
                img = cv2.imread(img)
            else:
                raise FileExistsError(img)
        return img

    @abstractmethod
    def test_single_image(self, img: Union[str, np.ndarray], ground_truth: int = None):
        pass

    @abstractmethod
    def pre_process_image(self, img: np.ndarray, crop_size: tuple[int, int] = (32, 32), device: str = 'cpu') -> torch.tensor:
        # Implement pre_process_image
        pass