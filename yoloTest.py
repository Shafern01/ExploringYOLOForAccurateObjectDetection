# Import necessary libraries
import torch
import cv2
from ultralytics import YOLO
import kagglehub

# Download latest version
path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")

print("Path to dataset files:", path)


# Force CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model and move to the selected device
model = YOLO("yolov8s.pt").to(device)  # Load model and set to use CUDA if available

# Initialize the computer camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera, you may need to change this index to 1 or 2 to find your camera
# Real-time object detection
try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 model on the captured frame
        results = model.predict(frame, device=device)  # Force prediction to use the CUDA device

        # Process the results
        for result in results[0].boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())  # Convert to int list
            class_id = int(result.cls.item())
            confidence = float(result.conf.item())
            label = model.names[class_id]

            # Draw bounding boxes and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bounding box
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Label

        # Display the resulting frame in an OpenCV window
        cv2.imshow("YOLOv8 Real-Time Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the camera and clear windows
    cap.release()
    cv2.destroyAllWindows()
