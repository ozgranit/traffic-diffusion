import os
from PIL import Image
from matplotlib import pyplot as plt


def plot_2_images_in_a_row(image1, image2, title1: str="", title2: str="", cmap=None, figsize=(10, 5), save_path=None, plot=True):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot the first image on the first subplot
    axes[0].imshow(image1, cmap=cmap)  # Use cmap='gray' for grayscale images_orig
    axes[0].set_title(title1)

    # Plot the second image on the second subplot
    axes[1].imshow(image2, cmap=cmap)  # Use cmap='gray' for grayscale images_orig
    axes[1].set_title(title2)

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  # Save the plot as an image
        print(f"Plot saved as {save_path}")
    if plot:
        plt.show()

# Define the root directory

# Create a function to find paired images and create plots
def __create_pair_plots_helper(road_folder: str):
    normal_path = os.path.join(road_folder, "normal_attack")
    special_path = os.path.join(road_folder, "special_attack")
    output_folder = os.path.join(road_folder, "plot_combined")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of image files in both folders
    normal_images = [f for f in os.listdir(normal_path) if f.endswith(".png")]
    special_images = [f for f in os.listdir(special_path) if f.endswith(".png")]

    # Loop through the images and create pair plots
    for image_file in normal_images:
        if image_file in special_images:
            normal_image_path = os.path.join(normal_path, image_file)
            special_image_path = os.path.join(special_path, image_file)

            # Open the images using Pillow
            normal_image = Image.open(normal_image_path)
            special_image = Image.open(special_image_path)

            # Create a pair plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(normal_image)
            axes[0].set_title("Normal Attack")
            axes[1].imshow(special_image)
            axes[1].set_title("Special Attack")
            plt.tight_layout()

            # Save the pair plot
            output_file = os.path.join(output_folder, f"pair_plot_{image_file}")
            plt.savefig(output_file)
            plt.close()

def create_pair_plots(root_dir: str):
    # Loop through the "road_x" folders
    for road_folder in os.listdir(root_dir):
        if not road_folder.startswith("road_"):
            continue
        road_path = os.path.join(root_dir, road_folder)
        if os.path.isdir(road_path):
            __create_pair_plots_helper(road_path)
    print("Pair plots created successfully.")

def plot_image(img, title: str =""):
    plt.imshow(img)
    plt.title(title)
    plt.show()