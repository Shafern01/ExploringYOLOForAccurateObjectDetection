from ultralytics import YOLO

def train_model(data_yaml_path, epochs, batch_size, img_size):
    print("Initializing YOLOv8m model training with tuned alpha (learning rate).")

    # Use YOLOv8m for better accuracy
    model = YOLO('yolov8m.pt')

    # Training configuration
    model.train(
        data=data_yaml_path,
        epochs=epochs,  # Total epochs
        batch=batch_size,  # Batch size
        imgsz=img_size,  # Image size
        save=True,  # Save checkpoints
        save_period=5,  # Save checkpoints every 5 epochs
        patience=7,  # Early stopping patience
        lr0=0.01,  # Tuned alpha (initial learning rate)
        lrf=0.05,  # Final learning rate for cosine decay
        momentum=0.9,  # Momentum for stability
        optimizer='AdamW',  # Use AdamW optimizer for better generalization
        augment=True,  # Enable advanced augmentations
        mosaic=True,  # Mosaic augmentation
        mixup=True,  # Mixup augmentation
        degrees=10,  # Random rotation augmentation
        translate=0.1,  # Random translation augmentation
        scale=0.5,  # Random scaling augmentation
        shear=2.0,  # Shearing augmentation
        verbose=True
    )

    print("Training completed successfully.")


if __name__ == '__main__':
    # Update paths and parameters for better accuracy
    data_yaml_path = r"C:\school\ML project files\yoloTestCharm\data.yaml"
    train_model(data_yaml_path, epochs=40, batch_size=16, img_size=640)  # Adjust batch size based on GPU memory
