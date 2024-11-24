from ultralytics import YOLO
import torch
import yaml
import os


def load_weights(data_yaml_path):
    """
    Loads class weights from data.yaml file
    """
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('weights', {})


def train_model(data_yaml_path, epochs, batch_size, img_size, validate_interval=10, patience=5):

    print("Initializing YOLOv8 model training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # GPU Optimization
    if device == 'cuda':
        print("Optimizing GPU settings...")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load class weights
    class_weights = load_weights(data_yaml_path)
    if class_weights:
        print("Loaded class weights from data.yaml")
    else:
        print("Warning: No class weights found in data.yaml")

    # Create output directory
    output_dir = "runs/train"
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model - using small model for 4060
    model = YOLO('yolov8s.pt')  # Using small model to fit in 8GB VRAM

    # Training tracking
    best_val_loss = float('inf')
    no_improvement_count = 0

    # Warmup epochs calculation
    warmup_epochs = max(3, int(epochs * 0.1))
    print(f"Using {warmup_epochs} warmup epochs")

    for epoch in range(1, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}...")

        # Dynamic learning rate
        if epoch <= warmup_epochs:
            current_lr = 0.001
        else:
            current_lr = 0.002

        # Train for one epoch
        results = model.train(
            data=data_yaml_path,
            epochs=1,
            batch=batch_size,
            imgsz=img_size,
            save=False,
            lr0=current_lr,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=warmup_epochs,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            optimizer='AdamW',
            cos_lr=True,
            augment=True,
            mosaic=1.0,
            mixup=0.2,
            degrees=5.0,
            translate=0.2,
            scale=0.2,
            shear=0.2,
            perspective=0.0,
            flipud=0.1,
            fliplr=0.5,
            verbose=True,
            amp=True,
            fraction=1.0,
            workers=4,  # Reduced workers for 16GB RAM
            device=device,
            project=output_dir,
            name=f'exp_epoch_{epoch}',
            exist_ok=True,
            pretrained=True,
            cache=True,
            close_mosaic=10,
            plots=True,
        )

        # Perform validation
        if epoch % validate_interval == 0:
            print(f"\nValidating model after epoch {epoch}...")
            val_results = model.val(
                data=data_yaml_path,
                batch=batch_size // 2,  # Smaller batch size for validation
                imgsz=img_size,
            )

            val_loss = val_results.box_loss + val_results.cls_loss + val_results.dfl_loss

            print(f"Validation Loss: {val_loss:.5f}")
            print(f"Metrics: mAP50={val_results.maps[0]:.3f}, mAP50-95={val_results.maps[1]:.3f}")

            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                best_model_path = os.path.join(output_dir, f'best_model_epoch_{epoch}.pt')
                model.save(best_model_path)
                print(f"New best model saved: {best_model_path}")
            else:
                no_improvement_count += 1
                print(f"No improvement in validation loss. Count: {no_improvement_count}/{patience}")

            if no_improvement_count >= patience:
                print("\nEarly stopping triggered. Training terminated.")
                break

    print("\nTraining completed successfully!")
    return model


if __name__ == '__main__':
    # Configuration optimized for RTX 4060 8GB
    config = {
        'data_yaml_path': r"C:\school\ML project files\yoloTestCharm\data.yaml",
        'epochs': 40,  # Reduced epochs for faster training
        'batch_size': 42,
        'img_size': 512,  # Balanced size for 4060
        'validate_interval': 5,
        'patience': 8,
    }

    # Train the model
    trained_model = train_model(**config)