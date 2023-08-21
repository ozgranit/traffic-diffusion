import os
import cv2
import xml.etree.ElementTree as ET

def get_coordinates(image):
    coordinates = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    # Display the image and wait for user clicks
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", on_click)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return coordinates

def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")

    annotations = []
    for obj in objects:
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        label = obj.find("name").text
        annotations.append((xmin, ymin, xmax, ymax, label))

    return annotations

def create_annotations(image_name, image_shape, annotations, xml_path = 'larger_images/image_annotations/'):
    tree = ET.parse(xml_path + 'road_1.xml')
    root = tree.getroot()
    name = root.find("filename")
    name.text = image_name
    size = root.find("size")
    size.find("width").text = str(image_shape[1])
    size.find("height").text = str(image_shape[0])

    objects = root.findall("object")

    for obj in objects:
        bbox = obj.find("bndbox")
        bbox.find("xmin").text = str(annotations['xmin'])
        bbox.find("ymin").text = str(annotations['ymin'])
        bbox.find("xmax").text = str(annotations['xmax'])
        bbox.find("ymax").text = str(annotations['ymax'])

    tree.write(xml_path + image_name.split('.')[0] + '.xml')

def process_image(images_folder, annotation_folder):
    file_names = []
    orig_imgs = []
    cropped_imgs = []
    cropped_resized_imgs = []
    labels = []

    for image_filename in os.listdir(images_folder):
        if image_filename.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(images_folder, image_filename)
            image = cv2.imread(image_path)

            # Collect coordinates from user clicks
            annotated_image = image.copy()
            coordinates = get_coordinates(annotated_image)

            # Ensure we have exactly 2 pairs of coordinates
            if len(coordinates) == 2:
                xmin, ymin = min(coordinates[0][0], coordinates[1][0]), min(coordinates[0][1], coordinates[1][1])
                xmax, ymax = max(coordinates[0][0], coordinates[1][0]), max(coordinates[0][1], coordinates[1][1])

                # Create a dictionary with the coordinates
                annotation = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                }
                print(f"Annotation for {image_filename}: {annotation}")
                create_annotations(image_filename, image.shape, annotation)
                # Display the annotated image
                # cv2.imshow("Annotated Image", annotated_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            else:
                print(f"Please select exactly 2 points on {image_filename}.")

            input("Press Enter to continue...")

if __name__ == "__main__":
    images_folder = 'larger_images/image_inputs'
    annotation_folder = 'larger_images/image_annotations'
    process_image(images_folder, annotation_folder)