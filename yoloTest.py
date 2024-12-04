import os
import cv2
import json
from ultralytics import YOLO
from yolo_utils import calculate_iou

"""
This script evaluates the YOLO model on local images and compares predictions
with real-time detections. It saves results for accuracy analysis and provides
a mechanism to calculate IoU-based accuracy.
"""


def get_detections(model, image_path):
    """
    Runs YOLO inference on a single image and extracts detections.
    Args:
        model (YOLO): The YOLO model object.
        image_path (str): Path to the input image.
    Returns:
        list: List of detections with 'class', 'confidence', and 'box'.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    predictions = model.predict(source=img)
    detections = []

    for result in predictions:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, class_ids):
            detections.append({
                'class': model.names[int(cls_id)],
                'confidence': float(conf),
                'box': box.tolist()
            })
    return detections


def compare_detections(detections1, detections2):
    """
    Compares two sets of detections using IoU.
    Args:
        detections1 (list): First set of detections (e.g., local image).
        detections2 (list): Second set of detections (e.g., live frame).
    Returns:
        float: Accuracy as the proportion of matched detections.
    """
    matched_detections = 0
    total_detections = len(detections1)

    for det1 in detections1:
        for det2 in detections2:
            iou = calculate_iou(det1['box'], det2['box'])
            if det1['class'] == det2['class'] and iou > 0.5:
                matched_detections += 1
                break

    accuracy = matched_detections / total_detections if total_detections > 0 else 0
    return accuracy


def save_comparison_results(results, save_path="C:/school/ML project files/yoloTestCharm/comparison_results.json"):
    """
    Saves the comparison results to a JSON file.
    Args:
        results (dict): Dictionary containing accuracy results.
        save_path (str): Path to save the JSON file.
    """
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Comparison results saved to {save_path}")


if __name__ == "__main__":
    # Paths
    model_path = r"C:\school\ML project files\yoloTestCharm\runs\train7\train\weights\best.pt"
    local_images_path = r"C:\school\ML project files\yoloTestCharm\ourTest_Images"
    live_save_dir = r"C:\school\ML project files\yoloTestCharm/live_frames"
    comparison_results_path = r"C:\school\ML project files\yoloTestCharm/comparison_results.json"

    # Load the YOLO model
    model = YOLO(model_path)

    # Run predictions on local images
    local_detections = {}
    local_image_paths = [
        os.path.join(local_images_path, file)
        for file in os.listdir(local_images_path)
        if file.endswith(".jpg")
    ]
    for image_path in local_image_paths:
        filename = os.path.basename(image_path)
        detections = get_detections(model, image_path)
        local_detections[filename] = detections

    # Capture live frames and compare
    live_image_paths = capture_live_frames(num_frames=50, save_dir=live_save_dir)
    comparison_results = {}
    for live_image_path in live_image_paths:
        live_filename = os.path.basename(live_image_path)
        live_detections = get_detections(model, live_image_path)
        corresponding_local_filename = live_filename.replace("live_frame_", "local_image_")
        local_detections_list = local_detections.get(corresponding_local_filename, [])
        accuracy = compare_detections(live_detections, local_detections_list)
        comparison_results[live_filename] = {"accuracy": accuracy}
    save_comparison_results(comparison_results)
