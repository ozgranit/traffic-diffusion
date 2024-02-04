import cv2

from inferenceAndResults.inference_on_src_attacked import load_model_and_set_true_label
from settings import LISA


def inference_single_image(img_path: str = None):
    image_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/output_api_road_1/snow_2.jpg'
    image_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/input_large_src_with_shadow/road_1.jpg'
    image_path = r'/workspace/traffic-diffusion/datasets/larger_images/image_inputs/road_1.jpg'
    image_path = image_path if img_path is None else img_path
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    crop_size = 32
    model_wrapper, true_label = load_model_and_set_true_label(adv_model=False, attack_db=LISA,
                                                              crop_size=crop_size)
    confidence, pred_label, attack_failed, msg = model_wrapper.test_single_image(image_path, true_label,
                                                                                 print_results=False)
    print(f"image_pat: {image_path}\npred_label: {pred_label}\nconfidence: {confidence}\nattack_failed: {attack_failed}\nmsg: {msg}")


if __name__ == "__main__":
    # img_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/output_api_road_1/snow_2.jpg'
    # inference_single_image(img_path)
    # print('-' * 30)
    # img_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/output_api_road_1_with_attack/snow_2.jpg'
    # inference_single_image(img_path)

    img_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/output_apidawn_2_rfla_normal/snow_2.jpg'
    img_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/output_apidawn_2_rfla_normal/midday_2.jpg'
    inference_single_image(img_path)

