# Import required libraries
import os  # For file and directory operations
import cv2  # For image processing and camera operations
import json  # For saving results in JSON format
from collections import Counter  # For counting class occurrences
import statistics  # For statistical calculations
from ultralytics import YOLO  # For YOLO model operations

"""
Enhanced script for visualizing and evaluating YOLO model's cross-domain performance.
Draws and saves detection boxes on both local and live images for visual comparison.
"""


def draw_detections(image, detections, save_path):
    """
    Draws bounding boxes and labels on image and saves it.
    """
    # Create a copy of the image to avoid modifying the original
    annotated_img = image.copy()

    # Process each detection in the image
    for det in detections:
        # Convert box coordinates from float to integer for drawing
        x1, y1, x2, y2 = map(int, det['box'])

        # Draw green rectangle for bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create label text with class name and confidence score
        label = f"{det['class']}: {det['confidence']:.2f}"

        # Calculate label background dimensions
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw green background rectangle for label
        cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + label_w, y1), (0, 255, 0), -1)

        # Draw label text in black
        cv2.putText(annotated_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Save the annotated image to disk
    cv2.imwrite(save_path, annotated_img)
    return annotated_img


def get_detections(model, image_path, confidence_threshold=0.5):
    """
    Runs YOLO inference on a single image and extracts detections.
    """
    # Load image from disk
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run model prediction with confidence threshold
    predictions = model.predict(source=img, conf=confidence_threshold)[0]

    # Extract detection information into a list of dictionaries
    detections = [{
        'class': model.names[int(cls_id)],  # Get class name from model's class mapping
        'confidence': float(conf),  # Convert confidence to float
        'box': box.tolist(),  # Convert box coordinates to list
        'image_size': img.shape[:2]  # Store image dimensions
    } for box, conf, cls_id in zip(
        predictions.boxes.xyxy.cpu().numpy(),  # Box coordinates
        predictions.boxes.conf.cpu().numpy(),  # Confidence scores
        predictions.boxes.cls.cpu().numpy()  # Class IDs
    )]

    return detections, img


def process_local_images(model, input_dir, output_dir, confidence_threshold=0.5):
    """
    Process all local images and save annotated versions.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    local_detections = {}

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        # Check if file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Create full file paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"annotated_{filename}")

            print(f"Processing local image: {filename}")
            # Get detections and original image
            detections, img = get_detections(model, input_path, confidence_threshold)
            # Draw and save annotated image
            draw_detections(img, detections, output_path)

            # Store detections for later analysis
            local_detections[filename] = detections

    return local_detections


def capture_and_process_live_frames(model, num_frames, save_dir, confidence_threshold=0.5):
    """
    Capture live frames, run detection, and save annotated frames.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    live_detections = {}

    try:
        # Capture and process specified number of frames
        for i in range(num_frames):
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame {i + 1}")
                continue

            # Generate filename and save original frame
            frame_filename = f"live_frame_{i + 1}.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Process frame with YOLO model
            print(f"Processing live frame {i + 1}/{num_frames}")
            predictions = model.predict(source=frame, conf=confidence_threshold)[0]

            # Extract detection information
            detections = [{
                'class': model.names[int(cls_id)],
                'confidence': float(conf),
                'box': box.tolist(),
                'image_size': frame.shape[:2]
            } for box, conf, cls_id in zip(
                predictions.boxes.xyxy.cpu().numpy(),
                predictions.boxes.conf.cpu().numpy(),
                predictions.boxes.cls.cpu().numpy()
            )]

            # Save annotated version of the frame
            annotated_path = os.path.join(save_dir, f"annotated_{frame_filename}")
            draw_detections(frame, detections, annotated_path)

            # Store detections for analysis
            live_detections[frame_filename] = detections

    finally:
        # Always release camera when done
        cap.release()

    return live_detections


def calculate_cross_domain_metrics(local_detections, live_detections):
    """
    Calculate metrics comparing detections across domains.
    """
    # Initialize counters for class frequencies
    local_classes = Counter()
    live_classes = Counter()

    # Count occurrences of each class in local images
    for detections in local_detections.values():
        for det in detections:
            local_classes[det['class']] += 1

    # Count occurrences of each class in live frames
    for detections in live_detections.values():
        for det in detections:
            live_classes[det['class']] += 1

    # Calculate comparison metrics
    metrics = {
        # Distribution of classes in each domain
        'class_distribution': {
            'local': dict(local_classes),
            'live': dict(live_classes)
        },
        # Analysis of class presence across domains
        'class_overlap': {
            'classes_in_both': list(set(local_classes.keys()) & set(live_classes.keys())),
            'only_in_local': list(set(local_classes.keys()) - set(live_classes.keys())),
            'only_in_live': list(set(live_classes.keys()) - set(local_classes.keys()))
        },
        # Average number of detections per image
        'average_detections': {
            'local': sum(local_classes.values()) / len(local_detections) if local_detections else 0,
            'live': sum(live_classes.values()) / len(live_detections) if live_detections else 0
        }
    }

    return metrics


if __name__ == "__main__":
    # Define configuration settings
    config = {
        'model_path': r"C:\school\ML project files\yoloTestCharm\runs\train7\train\weights\best.pt",
        'local_images_path': r"C:\school\ML project files\yoloTestCharm\ourTest_Images",
        'local_output_dir': r"C:\school\ML project files\yoloTestCharm\annotated_local",
        'live_frames_dir': r"C:\school\ML project files\yoloTestCharm\annotated_live",
        'results_path': r"C:\school\ML project files\yoloTestCharm\cross_domain_results.json",
        'confidence_threshold': 0.5,
        'num_frames': 5
    }

    # Initialize YOLO model
    print("Loading YOLO model...")
    model = YOLO(config['model_path'])

    # Process local test images
    print("\nProcessing local images...")
    local_detections = process_local_images(
        model,
        config['local_images_path'],
        config['local_output_dir'],
        config['confidence_threshold']
    )

    # Capture and process live camera frames
    print("\nCapturing and processing live frames...")
    live_detections = capture_and_process_live_frames(
        model,
        config['num_frames'],
        config['live_frames_dir'],
        config['confidence_threshold']
    )

    # Calculate performance metrics
    print("\nCalculating cross-domain metrics...")
    metrics = calculate_cross_domain_metrics(local_detections, live_detections)

    # Save all results to JSON file
    print(f"\nSaving results to {config['results_path']}...")
    results = {
        'metrics': metrics,
        'local_detections': local_detections,
        'live_detections': live_detections
    }

    with open(config['results_path'], 'w') as f:
        json.dump(results, f, indent=4)

    # Print completion message with output locations
    print("\nAnalysis complete!")
    print(f"- Annotated local images saved to: {config['local_output_dir']}")
    print(f"- Annotated live frames saved to: {config['live_frames_dir']}")
    print(f"- Analysis results saved to: {config['results_path']}")