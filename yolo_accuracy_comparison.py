import os
import json
import cv2
import torch
from ultralytics import YOLO

"""
This program integrates a YOLOv8 model for object detection and performs:
1. Static image inference: Evaluates test images against ground truth annotations, computes accuracy, and compares predictions.
2. Real-time detection: Captures live frames from a camera, processes detections, and saves annotated frames to a directory.

Features:
- IoU calculation for prediction-ground truth matching.
- Visualization of real-time detections with bounding boxes and confidence scores.
- Frame capture and storage for further analysis.
"""

# Initialize YOLO model
model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"
trained_model = YOLO(model_path)

# Paths
test_images_path = r"C:\Users\natha\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1\val\images"
ground_truth_path = r"C:\Users\natha\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1\val\annotations"
live_feed_save_path = r"C:\school\ML project files\yoloTestCharm\Captured live frames"

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0

    return iou


def load_ground_truth(annotations_path):
    ground_truth = {}
    for file in os.listdir(annotations_path):
        if file.endswith(".json"):
            with open(os.path.join(annotations_path, file), 'r') as f:
                data = json.load(f)

                # Check if 'data' is a list
                if isinstance(data, list):
                    for annotation in data:
                        image_name = annotation.get('name', None)
                        if image_name:
                            boxes = []
                            for label in annotation.get('labels', []):
                                if 'box2d' in label:
                                    box = label['box2d']
                                    boxes.append({
                                        'class': label['category'],
                                        'box': [box['x1'], box['y1'], box['x2'], box['y2']]
                                    })
                            ground_truth[image_name] = boxes
                else:
                    # If 'data' is a dictionary (original assumption)
                    image_name = data.get('name', None)
                    if image_name:
                        boxes = []
                        for label in data.get('labels', []):
                            if 'box2d' in label:
                                box = label['box2d']
                                boxes.append({
                                    'class': label['category'],
                                    'box': [box['x1'], box['y1'], box['x2'], box['y2']]
                                })
                        ground_truth[image_name] = boxes
    return ground_truth


# Compare predictions with ground truth
def compare_with_ground_truth(predictions, ground_truth):
    matched_detections = 0
    total_ground_truth = len(ground_truth)

    for gt in ground_truth:
        for pred in predictions:
            iou = calculate_iou(gt['box'], pred['box'])
            if gt['class'] == pred['class'] and iou > 0.5:
                matched_detections += 1
                break

    accuracy = matched_detections / total_ground_truth if total_ground_truth > 0 else 0
    return accuracy


# Run inference on static images
def static_inference(image_paths, ground_truth):
    results = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        predictions = trained_model.predict(source=img)
        detections = []

        # Extract bounding boxes, class IDs, and confidence scores
        for result in predictions:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates in [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                detections.append({
                    'class': trained_model.names[int(cls_id)],  # Map class ID to class name
                    'confidence': float(conf),
                    'box': box.tolist()  # Convert NumPy array to list
                })

        # Match ground truth to image filename
        filename = os.path.basename(image_path)
        gt = ground_truth.get(filename, [])

        # Compare with ground truth
        accuracy = compare_with_ground_truth(detections, gt)

        results.append({
            'filename': filename,
            'detections': detections,
            'ground_truth': gt,
            'accuracy': accuracy
        })
    return results


# Run real-time detection
def realtime_detection(frames_to_capture=50):
    cap = cv2.VideoCapture(0)
    results = []
    frame_count = 0

    os.makedirs(live_feed_save_path, exist_ok=True)

    while frame_count < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        predictions = trained_model.predict(source=frame)
        detections = []

        # Process YOLO predictions
        for result in predictions:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                detections.append({
                    'class': trained_model.names[int(cls_id)],  # Map class ID to class name
                    'confidence': float(conf),
                    'box': box.tolist()  # Convert NumPy array to list
                })

        # Save the frame to the live feed directory
        save_path = os.path.join(live_feed_save_path, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame)

        results.append({'frame': frame_count, 'detections': detections})

        # Visualize real-time detections
        for det in detections:
            box = det['box']
            label = det['class']
            confidence = det['confidence']
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return results


# Main program
if __name__ == "__main__":
    print("Loading ground truth annotations...")
    ground_truth = load_ground_truth(ground_truth_path)

    print("Collecting test image paths...")
    test_image_paths = [os.path.join(test_images_path, file) for file in os.listdir(test_images_path) if file.endswith(".jpg")]

    print("Running static inference on test images...")
    static_results = static_inference(test_image_paths, ground_truth)

    for result in static_results:
        print(f"Image: {result['filename']}, Accuracy: {result['accuracy']:.2f}")

    print("Running real-time detection...")
    realtime_results = realtime_detection()

    print("Real-time detection completed and frames saved at:", live_feed_save_path)
