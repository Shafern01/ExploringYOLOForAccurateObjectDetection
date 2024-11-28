# Standard library imports for deep learning and data processing
from ultralytics import YOLO  # YOLO implementation for object detection
import torch  # PyTorch deep learning framework
import yaml  # YAML file handling for configuration
import os  # Operating system interface
from pathlib import Path  # Object-oriented filesystem paths
from datetime import datetime  # Time handling
import matplotlib.pyplot as plt  # Plotting library
import json  # JSON handling for results
import numpy as np  # Numerical operations
import gc  # Garbage collection for memory management


def clear_system_cache():
    """
    Comprehensive memory cleanup optimized for RTX 4060 8GB GPU.
    This function ensures maximum available memory for training by:
    1. Clearing CUDA cache
    2. Resetting memory stats
    3. Setting memory allocation limits
    4. Optimizing CUDA settings for RTX cards
    """
    if torch.cuda.is_available():
        # Clear both CUDA cache types
        torch.cuda.empty_cache()  # Releases all unused cached memory
        torch.cuda.ipc_collect()  # Closes inter-process memory handles

        # Reset memory tracking statistics
        torch.cuda.reset_peak_memory_stats()  # Clears peak memory tracking
        torch.cuda.reset_accumulated_memory_stats()  # Resets accumulated stats

        # Set memory allocation strategy
        # 0.85 leaves 15% memory free for overhead and system operations
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.cuda.memory.empty_cache()

        # Enable TF32 for better performance on RTX cards
        # These settings can improve training speed by up to 20%
        torch.backends.cudnn.benchmark = True  # Enables cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enables TF32 for matrix multiplications
        torch.backends.cudnn.allow_tf32 = True  # Enables TF32 for convolutions

    # Force Python garbage collection
    gc.collect()


def get_next_train_number():
    """
    Manages training run numbering system to prevent overwriting previous results.

    Returns:
        int: Next available training number

    Logic:
    1. Checks 'runs' directory for existing training folders
    2. Extracts numbers from folder names
    3. Returns max number + 1 for next run
    """
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return 1

    # Get list of training directories
    existing_dirs = [d for d in os.listdir(runs_dir)
                     if os.path.isdir(os.path.join(runs_dir, d))
                     and d.startswith("train")]

    if not existing_dirs:
        return 1

    # Extract and find max training number
    numbers = [int(d.replace("train", "")) if d != "train" else 1
               for d in existing_dirs]

    return max(numbers) + 1


def plot_training_metrics(results_file, save_dir):
    """
    Creates comprehensive visualization of training metrics with advanced analytics.

    Args:
        results_file (str): Path to training results JSON
        save_dir (str): Directory to save plots

    Generates:
        2x2 grid of plots showing:
        1. Training losses with moving averages
        2. Validation metrics (mAP)
        3. Learning rate schedule
        4. Total loss convergence
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create 2x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Plot 1: Training Losses with moving average for smoothing
    epochs = list(range(1, len(results['train/box_loss']) + 1))
    window = 5  # Moving average window size for smoothing

    # Plot each loss type with its moving average
    for loss_type in ['box_loss', 'cls_loss', 'dfl_loss']:
        values = results[f'train/{loss_type}']
        # Calculate moving average for smoother visualization
        moving_avg = np.convolve(values, np.ones(window) / window, mode='valid')
        ax1.plot(epochs[window - 1:], moving_avg, label=f'{loss_type} (MA)')
        ax1.plot(epochs, values, alpha=0.3)  # Original values in lighter color

    ax1.set_title('Training Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Validation Metrics (mAP scores)
    # Filter out zero values which indicate no validation was performed
    val_epochs = [i for i, j in enumerate(results['metrics/mAP50(B)']) if j != 0]
    val_map50 = [j for j in results['metrics/mAP50(B)'] if j != 0]
    val_map50_95 = [j for j in results['metrics/mAP50-95(B)'] if j != 0]

    ax2.plot(val_epochs, val_map50, label='mAP50', marker='o')
    ax2.plot(val_epochs, val_map50_95, label='mAP50-95', marker='o')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Learning Rate Schedule
    ax3.plot(epochs, results['train/lr0'], label='LR')
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')  # Log scale for better visualization of LR decay
    ax3.grid(True)

    # Plot 4: Total Loss Convergence
    # Combine all losses to show overall training progress
    total_loss = np.array(results['train/box_loss']) + np.array(results['train/cls_loss']) + np.array(
        results['train/dfl_loss'])
    moving_avg_total = np.convolve(total_loss, np.ones(window) / window, mode='valid')
    ax4.plot(epochs[window - 1:], moving_avg_total, label='Total Loss (MA)')
    ax4.plot(epochs, total_loss, alpha=0.3)
    ax4.set_title('Loss Convergence')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Total Loss')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    plt.close()


def train_model(yolo_task, mode, data_yaml_path, epochs, batch_size, img_size, validate_interval=3, patience=25):
    """
    Main training function optimized for high mAP scores on BDD100k dataset.

    Args:
        yolo_task (str): Type of YOLO task ('detect')
        mode (str): Training mode
        data_yaml_path (str): Path to dataset configuration
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Input image size
        validate_interval (int): Epochs between validations
        patience (int): Early stopping patience

    Returns:
        model: Trained YOLO model
    """
    # Create unique training directory
    train_number = get_next_train_number()
    run_dir = os.path.join("runs", f"train{'' if train_number == 1 else train_number}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Training directory: {run_dir}")

    # Setup and optimize GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        clear_system_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(f"Total GPU memory: {gpu_memory:.2f} MB")

    # Create temporary YAML configuration
    temp_yaml_path = os.path.join(run_dir, 'temp_data.yaml')
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    # Initialize YOLOv8s model
    model = YOLO('yolov8s.pt')  # Using small model optimized for RTX 4060 8GB

    # Configure comprehensive training parameters
    train_args = dict(
        data=temp_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        save=True,
        save_period=validate_interval,
        patience=patience,
        device=device,
        project=run_dir,
        exist_ok=True,
        pretrained=True,
        amp=True,  # Automatic Mixed Precision
        verbose=True,
        cache=False,

        # Memory optimization parameters
        workers=2,  # Reduced workers for memory efficiency
        rect=True,  # Rectangular training
        profile=True,  # Performance profiling

        # Learning rate optimization
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # Weight decay for regularization
        warmup_epochs=5,  # Gradual warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation pipeline optimized for BDD100k
        augment=True,
        mosaic=0.9,  # Mosaic augmentation probability
        mixup=0.5,  # Mixup augmentation probability
        degrees=15.0,  # Maximum rotation degree
        translate=0.2,  # Translation factor
        scale=0.5,  # Scale factor
        shear=0.2,  # Shear factor
        perspective=0.0001,  # Perspective factor
        flipud=0.3,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,  # HSV saturation augmentation
        hsv_v=0.4,  # HSV value augmentation
        copy_paste=0.3,  # Copy-paste augmentation

        # Detection optimization
        conf=0.001,  # Confidence threshold
        iou=0.6,  # NMS IoU threshold
        single_cls=False,  # Multi-class detection

        # Loss weights
        box=1.0,  # Box loss weight
        cls=0.7,  # Class loss weight
        dfl=1.3,  # DFL loss weight

        # Advanced features
        close_mosaic=10,  # Disable mosaic in final epochs
        overlap_mask=True,  # Overlap mask inference
        cos_lr=True,  # Cosine learning rate scheduling
    )

    # Execute training with error handling
    try:
        results = model.train(**train_args)
        print(f"\nTraining completed! Results saved in: {run_dir}")

        # Generate training visualizations
        results_file = os.path.join(run_dir, 'results.json')
        if os.path.exists(results_file):
            plot_training_metrics(results_file, run_dir)
        else:
            print("Warning: Results file not found")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)
        clear_system_cache()

    return model


if __name__ == '__main__':
    # Configuration optimized for RTX 4060 8GB and high mAP goals
    config = {
        'yolo_task': 'detect',
        'mode': 'train',
        'data_yaml_path': r"C:\school\ML project files\yoloTestCharm\data.yaml",
        'epochs': 150,  # Extended training for better convergence
        'batch_size': 12,  # Optimized for 8GB VRAM
        'img_size': 640,  # Standard YOLO input size
        'validate_interval': 3,  # Frequent validation
        'patience': 25  # Extended patience for convergence
    }

    # Execute training
    trained_model = train_model(**config)