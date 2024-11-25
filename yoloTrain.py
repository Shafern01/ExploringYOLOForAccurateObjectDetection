from ultralytics import YOLO  # Import YOLO model framework
import torch  # PyTorch deep learning framework
import yaml  # For reading YAML configuration files
import os  # For file and directory operations
from pathlib import Path  # For platform-independent path handling
from datetime import datetime  # For timestamps


def get_next_train_number():
    """
    Determines the next available training run number by scanning existing directories.
    Returns:
        int: Next available training number (1 if no previous runs exist)
    """
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return 1

    # Get all directories that start with 'train'
    existing_dirs = [d for d in os.listdir(runs_dir)
                     if os.path.isdir(os.path.join(runs_dir, d))
                     and d.startswith("train")]

    if not existing_dirs:
        return 1

    # Extract numbers from directory names, handling 'train' (1) and 'trainX' cases
    numbers = [int(d.replace("train", "")) if d != "train" else 1
               for d in existing_dirs]

    return max(numbers) + 1


def train_model(data_yaml_path, epochs, batch_size, img_size, validate_interval=5, patience=15):
    """
    Main training function for YOLOv8 model with optimized parameters.

    Args:
        data_yaml_path (str): Path to data configuration file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
        validate_interval (int): Epochs between validations
        patience (int): Early stopping patience
    """
    # Set up training directory
    train_number = get_next_train_number()
    run_dir = os.path.join("runs", f"train{'' if train_number == 1 else train_number}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created training directory: {run_dir}")

    # Configure device and GPU settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        # Optimize GPU settings for training
        torch.cuda.empty_cache()  # Clear GPU memory
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN

    # Initialize model and training variables with optimized settings
    model = YOLO('yolov8s.pt')  # Load pre-trained YOLOv8 small model
    best_metric = 0  # Track best validation metric
    no_improvement_count = 0  # Counter for early stopping
    warmup_epochs = max(10, int(epochs * 0.15))  # Increased warmup period
    print(f"Using {warmup_epochs} warmup epochs")

    # Main training loop
    for epoch in range(1, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}...")

        # Create directory for this epoch
        epoch_dir = os.path.join(run_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Dynamic learning rate calculation with lower initial rate
        if epoch <= warmup_epochs:
            current_lr = 0.0003  # Lower initial learning rate for better stability
        else:
            # Gradually increase learning rate after warmup
            current_lr = min(0.002, 0.0003 + (0.002 - 0.0003) * (epoch - warmup_epochs) / (epochs - warmup_epochs))

        # Train for one epoch with optimized parameters
        results = model.train(
            # Dataset configuration
            data=data_yaml_path,
            epochs=1,
            batch=batch_size,
            imgsz=img_size,

            # Training optimization parameters
            save=False,  # Don't save intermediate checkpoints
            lr0=current_lr,  # Initial learning rate
            lrf=0.005,  # Final learning rate factor
            momentum=0.937,  # SGD momentum/Adam beta1
            weight_decay=0.001,  # Weight decay coefficient

            # Warmup configuration
            warmup_epochs=warmup_epochs,
            warmup_momentum=0.8,
            warmup_bias_lr=0.05,

            # Enhanced training techniques
            label_smoothing=0.1,  # Label smoothing epsilon
            dropout=0.2,  # Increased dropout for regularization

            # Optimizer settings
            optimizer='AdamW',  # Use AdamW optimizer
            cos_lr=True,  # Use cosine learning rate scheduler

            # Enhanced augmentation parameters
            augment=True,  # Enable augmentation
            mosaic=0.9,  # Increased mosaic probability
            mixup=0.4,  # Increased mixup probability
            degrees=15.0,  # More aggressive rotation
            translate=0.2,  # Translation range
            scale=0.5,  # More aggressive scaling
            shear=0.3,  # Shear range
            perspective=0.001,  # Perspective range
            flipud=0.3,  # Increased vertical flip probability
            fliplr=0.5,  # Horizontal flip probability

            # Hardware utilization
            verbose=True,  # Enable verbose output
            amp=True,  # Enable automatic mixed precision
            workers=2,  # Number of worker threads
            device=device,  # Training device

            # Output configuration
            project=run_dir,  # Project directory
            name=f'epoch_{epoch}',  # Run name
            exist_ok=True,  # Allow overwriting
            pretrained=True,  # Use pretrained weights
            cache=False,  # Disable caching
            close_mosaic=warmup_epochs,  # Disable mosaic after warmup
            plots=True,  # Generate plots
            rect=False,  # Disable rectangular training

            # Enhanced augmentation
            hsv_h=0.015,  # HSV hue augmentation
            hsv_s=0.7,  # HSV saturation augmentation
            hsv_v=0.4,  # HSV value augmentation
            copy_paste=0.3,  # Increased copy-paste augmentation

            # Multi-scale training for better scale invariance
            multi_scale=True
        )

        # Create weights directory for model checkpoints
        weights_dir = Path(os.path.join(epoch_dir, 'weights'))
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Validation phase
        if epoch % validate_interval == 0 or epoch == epochs:
            print(f"\nValidating model after epoch {epoch}...")
            val_results = model.val(
                data=data_yaml_path,
                batch=batch_size // 2,  # Smaller batch size for validation
                imgsz=img_size,
            )

            # Extract and display validation metrics
            val_map50 = val_results.maps[0]  # mAP at IoU 0.5
            val_map5095 = val_results.maps[1]  # mAP at IoU 0.5-0.95
            print(f"Validation Metrics: mAP@50={val_map50:.3f}, mAP@50-95={val_map5095:.3f}")

            # Model saving logic
            if val_map50 > best_metric:
                best_metric = val_map50
                no_improvement_count = 0
                best_model_path = os.path.join(weights_dir, f'best_model_epoch_{epoch}.pt')
                model.save(best_model_path)
                print(f"New best model saved: {best_model_path}")
            else:
                no_improvement_count += 1
                print(f"No improvement in validation mAP@50. Count: {no_improvement_count}/{patience}")

            # Early stopping check
            if no_improvement_count >= patience:
                print("\nEarly stopping triggered. Training terminated.")
                break

    print(f"\nTraining completed successfully! Results saved in: {run_dir}")
    return model


if __name__ == '__main__':
    # Optimized training configuration
    config = {
        'data_yaml_path': r"C:\school\ML project files\yoloTestCharm\data.yaml",  # Path to dataset config
        'epochs': 100,  # Doubled number of epochs
        'batch_size': 28,  # Reduced batch size for stability
        'img_size': 448,  # Input image size
        'validate_interval': 5,  # Validate every 5 epochs
        'patience': 15,  # Increased patience for better convergence
    }

    # Start training
    trained_model = train_model(**config)