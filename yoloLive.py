from yolo_utils import initialize_model, visualize_detections
import cv2
import os
import time
import json

"""
This script captures live frames from the laptop camera, performs real-time object detection using YOLO,
and saves the results (annotated frames and predictions) for further analysis.
"""

def live_detection(model_path, confidence_threshold=0.5, max_frames=50, save_dir="C:/school/ML project files/yoloTestCharm/live_detections"):
    """
    Performs real-time object detection using a webcam and YOLO model.
    Args:
        model_path (str): Path to YOLO model weights (.pt file).
        confidence_threshold (float): Minimum confidence for displaying detections.
        max_frames (int): Maximum number of frames to process.
        save_dir (str): Directory to save results (annotated frames and JSON predictions).
    """
    # Initialize the YOLO model
    model, device = initialize_model(model_path)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Prepare output directory
    os.makedirs(save_dir, exist_ok=True)

    frame_count = 0
    print("Press 'q' to quit.")

    # For measuring FPS
    start_time = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # Perform detection
        results = model.predict(source=frame, conf=confidence_threshold, device=device)
        detections = [
            {
                'box': box.xyxy[0].tolist(),
                'class': int(box.cls.item()),
                'confidence': float(box.conf.item())
            } for box in results[0].boxes
        ]

        # Save detections to a JSON file
        json_file_path = os.path.join(save_dir, f"frame_{frame_count}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(detections, json_file, indent=4)

        # Visualize detections
        visualize_detections(frame, detections, model.names)

        # Save the annotated frame
        annotated_frame_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(annotated_frame_path, frame)

        # Display the frame with detections
        cv2.imshow("YOLOv8 Real-Time Detection", frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection loop.")
            break

        frame_count += 1

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS).")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Path to the YOLO model weights
    model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"

    # Perform live detection
    live_detection(
        model_path=model_path,
        confidence_threshold=0.5,
        max_frames=50,
        save_dir="C:/school/ML project files/yoloTestCharm/live_detections"
    )
