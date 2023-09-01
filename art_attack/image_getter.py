import numpy as np
from PIL import Image
from art_attack import LISA_NUM_OF_CLASSES
from pathlib import Path


def load_larger_images():
    traffic_diffusion_dir = Path.cwd()
    # Load the image
    image_dir = traffic_diffusion_dir / 'larger_images' / 'image_inputs'

    # List of image paths
    image_paths = [image_dir / f'road_{i}.jpg' for i in range(1, 4)]

    # Load and reshape all images
    reshaped_images = [load_and_reshape_image(image_path) for image_path in image_paths]

    # Combine the reshaped images into a single NumPy array
    all_images = np.concatenate(reshaped_images, axis=0)

    # Print the shape of the resulting ndarray
    print("All images shape:", all_images.shape)

    # (800, 803, 3)
    # (800, 1155, 3)
    min_pixel_value = 0
    max_pixel_value = 255
    x_test = all_images
    y_test = get_stop_lisa_labels()
    x_train = None
    y_train = None

    return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value


def get_stop_lisa_labels(num_duplicates: int = 1) -> np.ndarray:
    stop_label_loc = 1
    label = [0] * LISA_NUM_OF_CLASSES
    label[stop_label_loc] = 1

    # Convert the list to a NumPy ndarray and repeat it
    duplicated_list = np.repeat(label, num_duplicates)

    # Reshape the ndarray to have the desired number of rows
    return duplicated_list.reshape(num_duplicates, len(label))


def load_and_reshape_image(image_path, target_size=(224, 224)):
    # Load the image
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize(target_size, Image.ANTIALIAS)

    # Convert the image to a NumPy ndarray
    image_np = np.array(resized_image)

    # Swap axes to PyTorch's NCHW format
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)  # Adding batch dimension

    return image_np

