import json
import os

# Mapping of categories to class ids (you can adjust this based on your dataset)
category_map = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9
}


def convert_to_yolo_format(json_data, image_width, image_height):
    annotations = []
    for label in json_data.get("labels", []):
        category = label["category"]
        if category in category_map:
            class_id = category_map[category]
            box = label["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            # Convert absolute coordinates to YOLO format (normalized)
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return annotations


def process_json_files(annotation_dir, image_dir, output_dir):
    for json_file in os.listdir(annotation_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(annotation_dir, json_file)
            with open(json_path, "r") as f:
                json_data = json.load(f)

            # Assuming json_data is a list of dictionaries
            for item in json_data:
                image_name = item["name"]
                image_path = os.path.join(image_dir, image_name)

                # Get image dimensions (you can use PIL or OpenCV to get these)
                from PIL import Image
                img = Image.open(image_path)
                img_width, img_height = img.size

                # Convert to YOLO format
                yolo_annotations = convert_to_yolo_format(item, img_width, img_height)

                # Write YOLO annotations to a text file
                txt_file_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
                with open(txt_file_path, "w") as txt_file:
                    for line in yolo_annotations:
                        txt_file.write(line + "\n")

# Set directories
annotation_dir = "C:/Users/natha/.cache/kagglehub/datasets/marquis03/bdd100k/versions/1/train/annotations"  # Directory containing JSON files
image_dir = "C:/Users/natha/.cache/kagglehub/datasets/marquis03/bdd100k/versions/1/train/images"  # Directory containing image files
output_dir = "C:/school/ML project files/yoloTestCharm/outputValAnnotation"  # Directory to store YOLO format .txt files

process_json_files(annotation_dir, image_dir, output_dir)
