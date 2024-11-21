from yolo_utils import initialize_model, visualize_detections
import cv2

def live_detection(model_path, confidence_threshold=0.5, max_frames=50):
    model, device = initialize_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    frame_count = 0
    print("Press 'q' to quit.")
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        results = model.predict(source=frame, conf=confidence_threshold)
        detections = [{
            'box': box.xyxy[0].tolist(),
            'class': int(box.cls.item()),
            'confidence': float(box.conf.item())
        } for box in results[0].boxes]
        visualize_detections(frame, detections, model.names)
        cv2.imshow("YOLOv8 Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = r"C:\school\ML project files\yoloTestCharm\runs\detect\train\weights\best.pt"
    live_detection(model_path)
