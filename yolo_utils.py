import torch
from ultralytics import YOLO
import os
import cv2

"""
This script contains utility functions for YOLO model initialization,
IoU calculation, and detection visualization. These utilities are designed
to be reused across various parts of the object detection pipeline.
"""


def initialize_model(model_path, device=None):
    """
    Initializes a YOLO model with the specified weights.
    Args:
        model_path (str): Path to the YOLO model weights (e.g., .pt file).
        device (str, optional): Device to use ('cuda' or 'cpu'). If None, it auto-detects.
    Returns:
        tuple: The YOLO model and the device being used.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        model = YOLO(model_path).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    return model, device


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1 (list): [x1, y1, x2, y2] coordinates of the first box.
        box2 (list): [x1, y1, x2, y2] coordinates of the second box.
    Returns:
        float: IoU value (0 <= IoU <= 1).
    """
    # Compute the intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # Compute the area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute the union area
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def visualize_detections(frame, detections, model_names):
    """
    Draws bounding boxes and labels on a given image frame.
    Args:
        frame (numpy.ndarray): The image frame to draw on.
        detections (list): List of detections, each containing 'box', 'class', and 'confidence'.
        model_names (list): List of class names corresponding to the model's class indices.
    """
    for det in detections:
        box = det['box']
        label = model_names[det['class']]
        confidence = det['confidence']

        # Draw bounding box
        cv2.rectangle(frame,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 255, 0), 2)
        # Draw label and confidence score
        cv2.putText(frame,
                    f"{label} {confidence:.2f}",
                    (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2)


# Example Usage (Optional)
if __name__ == "__main__":
    # Example of initializing a model
    model_path = "path/to/your/model.pt"
    model, device = initialize_model(model_path)

    # Example detections and visualization (replace with real data in practice)
    frame = cv2.imread("path/to/sample_image.jpg")
    example_detections = [
        {'box': [50, 50, 150, 150], 'class': 0, 'confidence': 0.95},
        {'box': [200, 200, 300, 300], 'class': 1, 'confidence': 0.89}
    ]
    model_names = ["person", "car"]  # Replace with actual model names
    visualize_detections(frame, example_detections, model_names)
    cv2.imshow("Detections", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
