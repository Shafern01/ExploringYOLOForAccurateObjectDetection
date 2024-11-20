import torch
from ultralytics import YOLO
import cv2
"""
This program performs object detection using a YOLOv8 model. It supports batch inference on a dataset 
and real-time detection via a connected camera. The program dynamically utilizes GPU (if available) 
for enhanced performance and provides bounding box annotations with class labels and confidence scores.
"""

if __name__ == '__main__':
    # Ensure CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLOv8 model
    model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"
    print("Loading the trained YOLO model for inference.")
    trained_model = YOLO(model_path).to(device)

    # Path to the dataset
    dataset_path = r"C:\Users\natha\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1\test"

    # Run inference in streaming mode
    print("Running inference on test images with streaming.")
    try:
        inference_results = trained_model.predict(
            source=dataset_path,
            conf=0.5,  # Confidence threshold
            save=True,  # Save results
            stream=True  # Stream mode to handle large datasets
        )

        # Process the results
        classes = trained_model.names  # Load class names
        for result in inference_results:
            boxes = result.boxes  # Bounding boxes
            for box in boxes:
                class_id = int(box.cls.item())
                label = classes[class_id]
                confidence = box.conf.item()
                bbox = box.xywh.tolist()  # Convert Tensor to list for safe formatting

                print(f"Class: {label}, Confidence: {confidence:.2f}")
                print(f"Bounding box: {bbox}")  # Center x, y, width, height

    except Exception as e:
        print(f"Error during inference: {e}")

    # Optional: Real-time object detection using the camera
    print("Initializing the camera for real-time detection.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera. Check the camera index.")
        exit()

    print("Starting real-time object detection. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting.")
                break

            # Run YOLOv8 model on the frame
            results = trained_model.predict(frame, device=device, conf=0.25)

            # Process the predictions
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())  # Safely convert Tensor to list
                class_id = int(result.cls.item())
                confidence = float(result.conf.item())
                label = trained_model.names[class_id]

                # Draw bounding boxes and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow("YOLOv8 Real-Time Detection", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    except Exception as e:
        print(f"Error during real-time detection: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
