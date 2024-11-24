import json
import os
from PIL import Image
from collections import Counter

# Mapping of categories to class IDs (matches your data.yaml)
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
    "traffic sign": 9,
    "lane": 10,
    "person": 11,
    "drivable area": 12,
    "bike": 13,
    "motor": 14
}


def load_fractional_dataset(fractional_images_path):
    """
    Loads the fractional dataset and returns a set of full image paths.

    Args:
        fractional_images_path (str): Path to the file containing fractional image paths

    Returns:
        set: Set of full image paths from the fractional dataset
    """
    if not os.path.exists(fractional_images_path):
        print(f"Error: Fractional images file not found at {fractional_images_path}")
        return set()

    with open(fractional_images_path, 'r') as f:
        # Read full paths from the file
        image_paths = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(image_paths)} images from fractional dataset")
    return image_paths


def get_image_name_from_path(full_path):
    """
    Extracts just the image name from a full path.

    Args:
        full_path (str): Full path to an image

    Returns:
        str: Just the image filename
    """
    return os.path.basename(full_path)


def convert_to_yolo_format(json_data, image_width, image_height):
    """Converts annotation data to YOLO format (keeping your existing implementation)"""
    # Your existing convert_to_yolo_format implementation remains the same
    annotations = []
    for label in json_data.get("labels", []):
        category = label["category"]
        if category in category_map:
            class_id = category_map[category]

            if "box2d" in label:
                box = label["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))

                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = max(0, (x2 - x1) / image_width)
                height = max(0, (y2 - y1) / image_height)

                if width > 0 and height > 0:
                    annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

            elif "poly2d" in label:
                for poly in label["poly2d"]:
                    vertices = poly["vertices"]
                    x_coords = [vertex[0] for vertex in vertices]
                    y_coords = [vertex[1] for vertex in vertices]

                    x_min = max(0, min(x_coords)) / image_width
                    y_min = max(0, min(y_coords)) / image_height
                    x_max = min(image_width, max(x_coords)) / image_width
                    y_max = min(image_height, max(y_coords)) / image_height

                    width = max(0, x_max - x_min)
                    height = max(0, y_max - y_min)
                    x_center = max(0, min(1, (x_min + x_max) / 2))
                    y_center = max(0, min(1, (y_min + y_max) / 2))

                    if width > 0 and height > 0:
                        annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return annotations


def process_json_files(annotation_dir, image_dir, output_dir, fractional_images_path):
    """
    Process JSON annotation files for images in the fractional dataset.

    Args:
        annotation_dir (str): Directory containing JSON annotation files
        image_dir (str): Directory containing images
        output_dir (str): Directory to save YOLO annotations
        fractional_images_path (str): Path to file containing fractional dataset image paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load the fractional dataset paths
    fractional_images = load_fractional_dataset(fractional_images_path)
    if not fractional_images:
        print("Error: No images loaded from fractional dataset")
        return

    # Create a set of just the image names from the fractional dataset
    fractional_image_names = {get_image_name_from_path(path) for path in fractional_images}

    processed_count = 0
    skipped_count = 0

    # Process each JSON file
    for json_file in os.listdir(annotation_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(annotation_dir, json_file)
        print(f"Processing {json_file}...")

        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse {json_file}")
            continue

        for item in json_data:
            image_name = item["name"]

            # Skip if image is not in fractional dataset
            if image_name not in fractional_image_names:
                skipped_count += 1
                continue

            # Get the actual image path from the fractional dataset
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error opening image {image_name}: {e}")
                continue

            # Convert annotations
            yolo_annotations = convert_to_yolo_format(item, img_width, img_height)

            # Save annotations
            base_name = os.path.splitext(image_name)[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")

            with open(txt_path, 'w') as f:
                for line in yolo_annotations:
                    f.write(f"{line}\n")

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")

    print(f"\nProcessing complete:")
    print(f"Total images processed: {processed_count}")
    print(f"Images skipped (not in fractional dataset): {skipped_count}")


if __name__ == "__main__":
    # Paths - using the same paths from your script
    annotation_dir = "C:/Users/natha/.cache/kagglehub/datasets/marquis03/bdd100k/versions/1/train/annotations"
    image_dir = "C:/Users/natha/.cache/kagglehub/datasets/marquis03/bdd100k/versions/1/train/images"
    output_dir = "C:/school/ML project files/yoloTestCharm/outputValAnnotation"
    fractional_images_path = "C:/school/ML project files/yoloTestCharm/fractional_images.txt"

    # Process annotations
    process_json_files(annotation_dir, image_dir, output_dir, fractional_images_path)

    # Validate results
    if os.path.exists(output_dir):
        annotation_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
        print(f"\nCreated {len(annotation_files)} annotation files")

        # Count classes
        class_counts = Counter()
        for ann_file in annotation_files:
            with open(os.path.join(output_dir, ann_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

        print("\nClass distribution in processed annotations:")
        for class_id, count in sorted(class_counts.items()):
            class_name = [k for k, v in category_map.items() if v == class_id][0]
            print(f"{class_name} (class {class_id}): {count} instances")