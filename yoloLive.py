import cv2
from ultralytics import YOLO
import torch

"""
yoloLive.py
This script runs a live feed from the camera and performs real-time object detection
using a YOLO model. Detected objects are displayed with bounding boxes, class labels,
and confidence scores in real-time.
"""

def run_live_feed(model_path, confidence_threshold=0.5):
    # Ensure CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path).to(device)

    # Initialize the camera
    print("Starting camera feed...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Run YOLO model on the current frame
            results = model.predict(source=frame, device=device, conf=confidence_threshold)

            # Process and visualize the detections
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                class_id = int(result.cls.item())
                confidence = float(result.conf.item())
                label = model.names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label and confidence
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("YOLOv8 Real-Time Detection", frame)

            # Quit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except Exception as e:
        print(f"Error during live detection: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path to the YOLO model
    model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"  # Change this to your desired YOLO model path (e.g., yolov8n.pt, yolov8s.pt, etc.)
    run_live_feed(model_path)

