from ultralytics import YOLO
from yolo_utils import initialize_model


def train_model(data_yaml_path, epochs, batch_size, img_size):
    print("Initializing YOLOv8 model training.")

    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Replace with other YOLO versions if needed.

    # Training with additional parameters
    model.train(
        data=data_yaml_path,  # Path to dataset YAML
        epochs=epochs,  # Total epochs
        batch=batch_size,  # Batch size
        imgsz=img_size,  # Image size
        save=True,  # Save checkpoints
        save_period=5,  # Save checkpoint every 5 epochs
        patience=5,  # Early stopping patience (stop if no improvement in 5 epochs)
        lr0=0.05,  # Initial learning rate
        lrf=0.1,  # Final learning rate for cosine scheduler
        optimizer='SGD',  # Optionally use AdamW or SGD
        verbose=True  # Show training progress in detail
    )

    print("Training completed successfully.")


if __name__ == '__main__':
    # Update these paths and parameters as needed
    data_yaml_path = r"C:\school\ML project files\yoloTestCharm\data.yaml"
    train_model(data_yaml_path, epochs=30, batch_size=32, img_size=320)
