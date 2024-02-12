import os


class Results:

    def __init__(self, model_name: str, experiment_folder: str, is_adv_model: bool):
        self.model_name = model_name
        self.experiment_folder = experiment_folder
        self.is_adv_model = is_adv_model
        self.total_images = 0
        self.total_src_images = 0
        self.total_diffusion_images = 0
        self.total_diffusion_imgs_attacked = 0
        self.total_diffusion_images_model_pred_correctly = 0
        self.model_pred_correctly_on_srcAdv_img = 0
        self.total_diff_imgs_with_at_lease_one_diffusion_image_success = 0
    def get_results(self) -> str:
        # Create the result string
        if self.total_diffusion_images == 0:
            result = (
                f"is_adv_model: {self.is_adv_model}\n"
                f"Total_images: {self.total_images}\n"
                f"Total_src_images: {self.total_src_images}\n"
                f"Total_diffusion_images: {self.total_diffusion_images}\n"
                f"model_pred_correctly_on_srcAdv_img: {self.model_pred_correctly_on_srcAdv_img}\n"
            )
        else:
            result = (
                f"is_adv_model: {self.is_adv_model}\n"
                f"Total_images: {self.total_images}\n"
                f"Total_src_images: {self.total_src_images}\n"
                f"Total_diffusion_images: {self.total_diffusion_images}\n"
                '***\n'
                f"Total_diffusion_imgs_attacked: {self.total_diffusion_imgs_attacked}, Acc: {(self.total_diffusion_imgs_attacked / self.total_diffusion_images) * 100}%\n"
                f"Total_diffusion_images_model_pred_correctly: {self.total_diffusion_images_model_pred_correctly}, Acc: {(self.total_diffusion_images_model_pred_correctly / self.total_diffusion_images) * 100}%\n"
                f"model_pred_correctly_on_srcAdv_img: {self.model_pred_correctly_on_srcAdv_img}\n"
                f"total_diff_imgs_with_at_lease_one_diffusion_image_success: {self.total_diff_imgs_with_at_lease_one_diffusion_image_success}\n"
        )

        result += '#' * 100
        result += '\n'

        return result

    def get_header(self) -> str:
        header = (
        f"model_name: {self.model_name}\n"
        f"experiment_folder: {self.experiment_folder}\n"
        )

        return header
    def save_and_display(self, dir_path: str, save_results: bool = True, save_to_file_type: str = 'w', is_adv_model: bool = False):
        results = self.get_results()
        if save_to_file_type == 'w':
            results = self.get_header() + results
        # Print the result
        print(results)
        if save_results:
            # Save the result to a file
            path = os.path.join(dir_path, 'inference.txt')
            with open(path, save_to_file_type) as f:
                f.write(f"Is adv model: {is_adv_model}")
                f.write(results)
