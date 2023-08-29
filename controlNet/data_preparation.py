import os
import random
import json
import shutil

# Define the source and target folders
source_folder = "larger_images/image_inputs"
target_folder = "larger_images/image_outputs"

# Load prompts from prompts.py
from prompts import prompt_getter

# Create the output folders
output_image = "controlNet/images_orig"
output_condition_images = "controlNet/conditioning_images_orig"
json_output = "controlNet/train.jsonl"
os.makedirs(output_image, exist_ok=True)
os.makedirs(output_condition_images, exist_ok=True)

# Initialize the data list for JSONL
data = []


# Function to randomly choose a prompt for a given target
def get_random_prompt(topic):
    prompts = prompt_getter.get(topic)
    prompts = prompts.split('.')
    if len(prompts[-1]) == 0:
        prompts = prompts[:-1]
    return random.choice(prompts)

cnt=0
# Process each target folder
for target_name in sorted(os.listdir(target_folder), key=lambda n: int(n.split('_')[1])):
    target_name_path = os.path.join(target_folder, target_name)
    if os.path.isdir(target_name_path):
        target_images = os.listdir(target_name_path)
        for image_name in target_images:
            if image_name.endswith(".jpg"):
                target_image_path = os.path.join(target_name_path, image_name)
                target_topic, extension = os.path.splitext(image_name)
                source_image_path = os.path.join(source_folder, f"{target_name}.jpg")
                source_conditioning_image_path = os.path.join(output_condition_images, f"{target_name}_{image_name}")
                image_output_path = os.path.join(output_image, f"{target_name}_{image_name}")

                # Copy the source image to conditioning_images_orig
                shutil.copy(source_image_path, source_conditioning_image_path)
                shutil.copy(target_image_path, image_output_path)

                # Get a random prompt for the target topic
                prompt = get_random_prompt(target_topic)

                # Create JSONL data entry
                data_entry = {
                    "text": prompt,
                    "image": f"{os.path.split(output_image)[1]}/{target_name}_{image_name}",
                    "conditioning_image": f"{os.path.split(output_condition_images)[1]}/{target_name}_{image_name}"
                }
                data.append(data_entry)
                cnt+=1

                # if cnt==3:
                #     break
    # if cnt == 3:
    #     break

# Write data to train.jsonl
with open(json_output, "w") as jsonl_file:
    for entry in data:
        jsonl_file.write(json.dumps(entry) + "\n")
