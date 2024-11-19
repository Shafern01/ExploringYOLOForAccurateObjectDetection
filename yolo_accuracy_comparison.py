import cv2
import torch
import os
from ultralytics import YOLO
from ultralytics.utils import bbox_iou

# Initialize YOLO model
model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"  # Update with your YOLO model path
trained_model = YOLO(model_path)

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    return bbox_iou(torch.tensor(box1), torch.tensor(box2)).item()

# Function to compare static inference results with real-time detection
def compare_results(static_results, realtime_results):
    comparisons = []
    for static_frame, realtime_frame in zip(static_results, realtime_results):
        static_detections = static_frame['detections']
        realtime_detections = realtime_frame['detections']

        matched_detections = 0
        total_static = len(static_detections)
        total_realtime = len(realtime_detections)

        for static_det in static_detections:
            for realtime_det in realtime_detections:
                iou = calculate_iou(static_det['box'], realtime_det['box'])
                if static_det['class'] == realtime_det['class'] and iou > 0.5:
                    matched_detections += 1
                    break

        comparisons.append({
            'frame': static_frame['frame'],
            'static_detections': total_static,
            'realtime_detections': total_realtime,
            'matched_detections': matched_detections
        })

    return comparisons

# Function to visualize detections
def visualize_detections(image, detections, title="Detection"):
    for det in detections:
        box = det['box']
        class_name = det['class']
        confidence = det['confidence']
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow(title, image)

# Run inference on static images
def static_inference(image_paths):
    results = []
    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        prediction = trained_model.predict(source=img, stream=True)
        detections = []
        for det in prediction:
            detections.append({
                'class': det['class'],
                'confidence': det['confidence'],
                'box': det['box']
            })
        results.append({'frame': idx, 'detections': detections, 'image': img})
    return results

# Run real-time detection
def realtime_detection(camera_id=0, frames_to_capture=50):
    cap = cv2.VideoCapture(camera_id)
    results = []
    frame_count = 0

    while frame_count < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        prediction = trained_model.predict(source=frame, stream=True)
        detections = []
        for det in prediction:
            detections.append({
                'class': det['class'],
                'confidence': det['confidence'],
                'box': det['box']
            })
        results.append({'frame': frame_count, 'detections': detections, 'image': frame})

        # Visualize real-time detections
        visualize_detections(frame.copy(), detections, title=f"Real-Time Frame {frame_count}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return results

# Compare detections with ground truth
def compare_with_ground_truth(detections, ground_truth):
    matched_detections = 0
    total_ground_truth = len(ground_truth)

    for gt in ground_truth:
        for det in detections:
            iou = calculate_iou(gt['box'], det['box'])
            if gt['class'] == det['class'] and iou > 0.5:
                matched_detections += 1
                break

    accuracy = matched_detections / total_ground_truth if total_ground_truth > 0 else 0
    return accuracy

# Main function
if __name__ == "__main__":
    # Dynamic loading of test image paths
    folder_path = r"C:\path\to\test\images"  # Update this path to your folder containing test images
    test_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(test_image_paths)} images in the folder.")

    # Ground truth data (example format)
    ground_truth_data = [
        {'class': 'car', 'box': [50, 50, 200, 200]},  # Example ground truth bounding box
        # Add ground truth data for each frame
    ]

    # Run static inference
    print("Running static inference on test images...")
    static_results = static_inference(test_image_paths)

    # Run real-time detection
    print("Running real-time detection...")
    realtime_results = realtime_detection()

    # Compare results
    print("Comparing results between static inference and real-time detection...")
    comparison_results = compare_results(static_results, realtime_results)

    # Evaluate against ground truth
    print("Evaluating accuracy against ground truth...")
    for idx, static_result in enumerate(static_results):
        frame_accuracy = compare_with_ground_truth(static_result['detections'], ground_truth_data)
        print(f"Frame {idx} Accuracy: {frame_accuracy:.2f}")

    # Visualize static inference results
    print("Visualizing static inference results...")
    for result in static_results:
        visualize_detections(result['image'], result['detections'], title=f"Static Frame {result['frame']}")
        cv2.waitKey(0)

    # Close all visualization windows
    cv2.destroyAllWindows()
