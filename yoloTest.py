import torch
import cv2
from ultralytics import YOLO
import kagglehub

# Main safeguard for multiprocessing
if __name__ == '__main__':
    # Ensure CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLOv8 model
    model = YOLO("yolov8s.pt").to(device)

    # Train the model
    try:
        print("Training YOLO model...")
        results = model.train(data="C:/school/ML project files/yoloTestCharm/data.yaml", batch=16, epochs=10, imgsz=320)
        print("Training complete.")
    except Exception as e:
        print(f"Error during training: {e}")
        exit()

    # Load the trained YOLO model
    #print("Loading the trained YOLO model for inference.")
    #trained_model = YOLO("path/to/best.pt").to(device)  # Replace with the path to your weights file

    # Run inference
   # print("Running inference on test images.")
    #inference_results = trained_model.predict(source="path/to/inference/images", conf=0.5, save=True)

    # Display or process the inference results
    for result in inference_results:
        # Each 'result' object holds details for one image
        boxes = result.boxes  # Bounding boxes
        classes = result.names  # Class labels

        # Print each detected object with its coordinates and label
        for box in boxes:
            print(f"Class: {classes[int(box.cls)]}, Confidence: {box.conf:.2f}")
            print(f"Bounding box: {box.xywh}")  # Center x, y, width, height

    # Initialize the camera
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
            results = model.predict(frame, device=device, conf=0.25)  # Lower confidence threshold as needed

            # Process the predictions
            for result in results[0].boxes:
                # Extract bounding box info
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                class_id = int(result.cls.item())
                confidence = float(result.conf.item())
                label = model.names[class_id]

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
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
