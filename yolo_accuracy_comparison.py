# Import necessary libraries
from yolo_utils import initialize_model
import os
import cv2

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def batch_inference(model_path, dataset_path, save_dir, conf=0.5, save=True):
    """Run batch inference and save results."""
    model, device = initialize_model(model_path)
    results = model.predict(source=dataset_path, conf=conf, save=save, save_dir=save_dir, stream=True)
    return results

def static_inference(image_paths, ground_truth, model):
    """Run inference on images and compare with ground truth."""
    results = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        predictions = model.predict(source=img)
        detections = []
        for result in predictions:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                detections.append({'class': model.names[int(cls_id)], 'confidence': float(conf), 'box': box.tolist()})
        filename = os.path.basename(image_path)
        gt = ground_truth.get(filename, [])
        accuracy = compare_with_ground_truth(detections, gt)
        results.append({'filename': filename, 'detections': detections, 'ground_truth': gt, 'accuracy': accuracy})
    return results

def realtime_detection(model, frames_to_capture=50, save_dir="live_frames"):
    """Run real-time YOLO detection on live camera feed."""
    cap = cv2.VideoCapture(0)
    frame_count = 0
    os.makedirs(save_dir, exist_ok=True)
    while frame_count < frames_to_capture:
        ret, frame = cap.read()
        if not ret:
            break
        predictions = model.predict(source=frame)
        for result in predictions:
            for box, conf, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[int(cls_id)]} {conf:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("YOLO Real-Time Detection", frame)
        save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
