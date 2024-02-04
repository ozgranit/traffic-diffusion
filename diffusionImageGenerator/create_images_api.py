# https://docs.getimg.ai/operation/operation-post_stable_diffusion_xl_image_to_image
import json

import cv2
import base64
import numpy as np
import os
import requests
import random
from prompts import prompt_getter, NEGATIVE_PROMPT

TOKENS = ['key-31ffFs5sAm0frJBhYw3Otcw4PpyurGUZBcdzgbZfLS2E1VHb0tscb6MfX9aahh4srcNPIGi8W3qdFCkTonQQIevkADcDZIjo',
          'key-nW3E6RoU5x6r5Wf6SPXdalHFUrXP2eBXkb8kpL7WMpJc54JOTiT9UIE8Jph9HHB6EEfO6jTDvfYUYkBJpsSrnmUfTKEZuuO',
          'key-1ApfAS0MprJ98FJJaOUYUCeZwHsI5n9tcFJzNRiSm7cQN3CsyRYJ7rvIPtRnHEfCcG6ozNoesB7GW5AeWWuhVQQMMFfp5i0Z',
          'key-38Hgtd7b7rpE47jNGYi3T8JwnSvnSQp58qz2wtEvJtL3W3NAu7iVS5YKwwtg8hJKARjDVBaL0fGixxyV6MrbI2IbvKgIJO4s',
          'key-2qQwGmWcaXRF3ZneDpaTl6CzDrkoqSwicq3k2v94uoQRV9G5KDNa0GafE2BBlMDOsVU1qLXQJOGowWs9kQQjvbUpho4g7SBr',
          'key-1Z4hUUXwaT05HQP2hJeYxEcprTFMGDlYp06n2rAqDGhM3aaFDCReUbuq4o582e3GxcIhDcRQx1tyw9JALOoUQE2s344vm6JT',
          'key-3l9HXVU9Yhk1zNbni3tr8rxEPIcPnDFE5UVWFkfGnKJZpoFWrwl7ryM6ZIIDlhpHXoboRTSfoG80LrEOdbMlxmcw7E8JZduz',
          'key-2fKUAkv2lHMZ1X4GFW46w8NYLo68oagnVKFXvmRKJFiiIzSKzzDKFM4PxeuxwSP3VzMwxfIlGxl6giBSNN5f7HSwP3lVywYg',
          'key-4cIQGuaE1dhWK6mB1C8B3ZLh6sWno791RR6xFfnK3pZEZfOyLTefOPnzcJuWWWXQ1HWtTo6u4HqyiLIJpSNvb9gQ5MBtHBNy',
          'key-aaTpSoBjh6chL4H0Vbh7v3MxlciuRDtNZTvufHEy3LSHAIKxJ3zFnWQoQM0kgkj9kWU6UV8StDuY70fvotyxGQMV9SDXLIU',
          'key-3X8S5uwKONys5ybITKRFxNxqsca3db4JmneJgTRkr2tCj20NAkOC16HLkk5apkT8EcnFBVJWgM0RW0vEtYilBabI7am7bIWJ',
          'key-2vbI5KMRKfF0rWcMHLwGo3uMjz2xVx42dehXYVLHY49jueXuWBLledXDZYZomgKlI06v5miEkLrW1n2Jigz936chrbflOBpP']

URL_BASE = 'https://api.getimg.ai/v1'
URL_SUFFIX = '/stable-diffusion-xl/image-to-image'

OUTPUT_FOLDER = r'datasets/imags_with_shadow/output'    #'larger_images/image_outputs/'
INPUT_FOLDER = r'datasets/imags_with_shadow/input'  #'larger_images/image_inputs/'


def load_images(folder_path, images_filter=None):
    if images_filter is None:
        images_filter = []
    original_images = {}
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        image_num = int(filename.split('.')[0].split('_')[1])
        if img is not None and image_num not in images_filter:
            original_images[filename.split('.')[0]] = img
    return original_images


def generate_images(original_images):
    url = URL_BASE + URL_SUFFIX
    total_cost = 0

    # replace with output_folder = OUTPUT_FOLDER, just so we won't overwrite the original images_orig
    output_folder = 'larger_images/image_outputs/'
    os.makedirs(output_folder, exist_ok=True)

    # strength -    Indicates how much to transform the reference image. When strength is 1, initial image will be
    #               ignored. Technically, strength parameter indicates how much noise add to the image.
    #               Minimum value is 0, maximum value is 1. Default value is 0.5.
    # steps -       The number of denoising steps. More steps usually can produce higher quality images_orig,
    #               but take more time to generate. Number of steps is modulated by strength.
    #               Notice that more steps = more cost, so don't abuse it.
    #               Minimum value is 1, maximum value is 100. Default value is 50
    # guidance -    Guidance scale as defined in Classifier-Free Diffusion Guidance.
    #               Higher guidance forces the model to better follow the prompt, but result in lower quality output.
    #               Minimum value is 0, maximum value is 20. Default value is 7.5.
    data = {
        "model": "stable-diffusion-v1-5",
        "prompt": "a photo of an astronaut riding a horse on mars",
        "negative_prompt": "",
        "image": "...looooong base64 encoded image string...",
        "strength": 0.7,
        "steps": 30,
        "guidance": 20,
        "seed": 42,
        "scheduler": "euler",
        "output_format": "jpeg"
    }
    headers = {
        "Authorization": "Bearer Your_Auth_Token",
        "Content-Type": "application/json"
    }

    max_retries = 5

    breaker = False
    received_images = []
    for index, (filename, image) in enumerate(original_images.items()):
        for prompt_desc, cur_prompt in prompt_getter.items():
            retry_count = 0
            while retry_count < max_retries:
                seed = random.randint(1, 2147483647)
                bearer_token = TOKENS[0]

                # convert the image to base64 encoding
                _, buffer = cv2.imencode('.jpg', image)
                base64_image = base64.b64encode(buffer).decode('utf-8')

                data['model'] = 'stable-diffusion-xl-v1-0'
                data['image'] = base64_image
                data['prompt'] = cur_prompt
                data['negative_prompt'] = NEGATIVE_PROMPT
                data['strength'] = 0.7
                data['steps'] = 10
                data['guidance'] = 17
                data['seed'] = seed
                headers['Authorization'] = 'Bearer ' + bearer_token

                response = requests.post(url, json=data, headers=headers)
                if response.status_code == 200:
                    cur_output_folder = output_folder + filename
                    os.makedirs(cur_output_folder, exist_ok=True)
                    cur_output_folder += '/'
                    output_filename = prompt_desc + '_2.jpg'

                    response_json = response.json()

                    # Extract the base64 encoded image string from the JSON response
                    base64_image_string = response_json["image"]

                    # decode the base64 image string to bytes
                    image_bytes = base64.b64decode(base64_image_string)

                    # convert the image bytes to a np array and decode into an OpenCV image
                    image_array = np.frombuffer(image_bytes, np.uint8)
                    cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    received_images.append(cv_image)
                    cv2.imwrite(cur_output_folder + output_filename, cv_image)

                    # cv2.imshow(filename, cv_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    print('Filename:', filename, 'Prompt:', prompt_desc, 'Cost:', response_json['cost'])
                    total_cost += response_json['cost']
                    print('Current cost:', response_json['cost'], 'Total cost:', total_cost)
                    break
                else:
                    retry_count += 1
                    error_json = json.loads(response.text)
                    print("Request failed:", error_json['error']['code'])
                    if error_json['error']['code'] == 'quota_exceeded':
                        print('Trying the next token...')
                        TOKENS.pop(0)
                    if len(TOKENS) == 0:
                        print('No more tokens to try, exiting...')
                        breaker = True
                        break
            if breaker:
                break
        if breaker:
            break


if __name__ == "__main__":
    # images_to_filter = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    images_to_filter = []
    original_images = load_images(INPUT_FOLDER, images_to_filter)
    generate_images(original_images)
