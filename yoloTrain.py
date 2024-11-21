from yolo_utils import initialize_model
from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, batch_size=16, img_size=640):
    print("Initializing YOLOv8 model training.")
    model = YOLO('yolov8n.pt')  # You can change the architecture as needed.
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size
    )
    print("Training completed successfully.")

if __name__ == '__main__':
    data_yaml_path = r"C:\school\ML project files\yoloTestCharm\data.yaml"
    train_model(data_yaml_path)