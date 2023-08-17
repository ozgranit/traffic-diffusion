import json
import os
import pandas as pd
from classification.utils import Utils

IMAGES_PATH = 'kaggle_images/'
OUTPUT = ''

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
    df.to_csv(OUTPUT + 'classification_results.csv', index=False)