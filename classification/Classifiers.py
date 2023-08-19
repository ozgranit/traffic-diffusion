import json
import os
import pandas as pd
from model_utils import Utils

from model_utils import MODELS_PATH

IMAGES_PATH = 'kaggle_images/'
OUTPUT_FOLDER = ''

if __name__ == '__main__':
    results = []

    utils = Utils()
    for filename in os.listdir(IMAGES_PATH):
        if filename.lower().endswith(('.png', '.jpg')):
            results.append(utils.test_single_image(IMAGES_PATH, filename, 'GTSRB', adv_model=False))
            results.append(utils.test_single_image(IMAGES_PATH, filename, 'GTSRB', adv_model=True))
            results.append(utils.test_single_image(IMAGES_PATH, filename, 'Lisa', adv_model=False))
            results.append(utils.test_single_image(IMAGES_PATH, filename, 'Lisa', adv_model=True))

    df = pd.DataFrame(results)
    output_path = OUTPUT_FOLDER + 'classification_results.csv'
    df.to_csv(output_path, index=False)
    print("Results save to ", output_path)
    print("Done!")