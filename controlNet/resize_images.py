import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Input and output directories
input_folder = "controlNet/conditioning_images_orig"
output_folder = "controlNet/conditioning_images"
os.makedirs(output_folder, exist_ok=True)

# Target size
target_size = (512, 512)

# Iterate through images_orig in the input folder
for image_name in os.listdir(input_folder):
    if image_name.endswith((".jpg", ".png")):
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        # Open the image using PIL
        image = Image.open(image_path)

        # Calculate the aspect ratio
        width, height = image.size
        aspect_ratio = width / height

        # Calculate new size while maintaining the aspect ratio
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        # Resize the image using torch interpolation
        image_tensor = TF.to_tensor(image)
        resized_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False)
        resized_image = TF.to_pil_image(resized_tensor.squeeze())

        # Save the resized image
        resized_image.save(output_path)

print("Images resized and saved to the output folder.")
