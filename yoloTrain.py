# Standard library imports
from ultralytics import YOLO  # Import YOLO implementation for object detection
import torch  # PyTorch deep learning framework
import yaml  # YAML file handling for configuration
import os  # Operating system interface for file operations
from pathlib import Path  # Object-oriented filesystem paths
from datetime import datetime  # Time handling utilities
import matplotlib.pyplot as plt  # Plotting library for visualizations
import json  # JSON handling for reading training results
import numpy as np  # Numerical operations library
import gc  # Garbage collection for memory management


def clear_system_cache():
    """
    Comprehensive function to clear various types of cache before training.
    Optimized specifically for RTX 4060 8GB GPU.

    This function performs several critical memory management tasks:
    1. Clears GPU memory caches
    2. Optimizes CUDA memory settings
    3. Enables RTX-specific optimizations
    4. Performs garbage collection
    """
    # Clear PyTorch cache if GPU is available
    if torch.cuda.is_available():
        # Clear both primary and secondary CUDA caches
        torch.cuda.empty_cache()  # Releases unused cached memory
        torch.cuda.ipc_collect()  # Cleans inter-process memory handles

        # Reset all memory tracking statistics
        torch.cuda.reset_peak_memory_stats()  # Clears peak memory tracking
        torch.cuda.reset_accumulated_memory_stats()  # Resets accumulated stats

        # Set optimal memory allocation strategy for 8GB VRAM
        torch.cuda.set_per_process_memory_fraction(0.85)  # Reserve 15% for system operations
        torch.cuda.memory.empty_cache()  # Additional cache clearing

        # Enable RTX optimizations
        torch.backends.cudnn.benchmark = True  # Enables cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enables TF32 for matrix operations
        torch.backends.cudnn.allow_tf32 = True  # Enables TF32 for convolutions

    # Force Python garbage collection
    gc.collect()


def get_next_train_number():
    """
    Manages training run numbering by scanning existing directories.
    Ensures each training run has a unique sequential number.

    Returns:
        int: Next available training number starting from 1

    This function:
    1. Checks 'runs' directory for existing training folders
    2. Extracts numbers from existing folder names
    3. Returns the next number in sequence
    """
    # Define the base directory for training runs
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return 1

    # Get list of existing training directories
    existing_dirs = [d for d in os.listdir(runs_dir)
                     if os.path.isdir(os.path.join(runs_dir, d))
                     and d.startswith("train")]

    if not existing_dirs:
        return 1

    # Extract numbers from directory names and find the maximum
    numbers = [int(d.replace("train", "")) if d != "train" else 1
               for d in existing_dirs]

    return max(numbers) + 1


def plot_training_metrics(results_file, save_dir):
    """
    Creates comprehensive visualization of training metrics including losses and validation metrics.
    Generates enhanced visualizations with moving averages and detailed analysis.

    Args:
        results_file (str): Path to the JSON file containing training results
        save_dir (str): Directory where the plot will be saved

    Generates:
        - 2x2 grid of plots showing different aspects of the training process
        - Includes moving averages for smoother visualization
        - Saves high-resolution plot to disk
    """
    # Load training results from JSON file
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Plot 1: Training Losses with moving averages
    epochs = list(range(1, len(results['train/box_loss']) + 1))
    window = 5  # Window size for moving average calculation

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
    ax3.set_yscale('log')  # Log scale for better visualization
    ax3.grid(True)

    # Plot 4: Total Loss Convergence
    total_loss = np.array(results['train/box_loss']) + np.array(results['train/cls_loss']) + np.array(
        results['train/dfl_loss'])
    moving_avg_total = np.convolve(total_loss, np.ones(window) / window, mode='valid')
    ax4.plot(epochs[window - 1:], moving_avg_total, label='Total Loss (MA)')
    ax4.plot(epochs, total_loss, alpha=0.3)
    ax4.set_title('Loss Convergence')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Total Loss')
    ax4.grid(True)

    # Save the plot with high DPI
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    plt.close()


def train_model(yolo_task, mode, data_yaml_path, epochs, batch_size, img_size, validate_interval=3, patience=30):
    """
    Main training function optimized for high mAP scores on BDD100k dataset.
    Specifically tuned for RTX 4060 8GB GPU.

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
    print(f"Created training directory: {run_dir}")

    # Setup GPU/CPU device and optimize GPU settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        clear_system_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(f"Total GPU memory: {gpu_memory:.2f} MB")

    # Create temporary YAML file for training configuration
    temp_yaml_path = os.path.join(run_dir, 'temp_data.yaml')
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    # Initialize model with pre-trained weights
    model = YOLO('yolov8s.pt')

    # Configure training parameters
    train_args = dict(
        # Basic parameters
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
        amp=True,
        verbose=True,
        cache=False,

        # Memory optimization
        workers=2,
        rect=False,  # Disabled to allow shuffling
        profile=True,

        # Learning rate settings
        lr0=0.0007,  # Reduced initial learning rate
        lrf=0.0001,  # Modified final LR factor
        momentum=0.937,
        weight_decay=0.0007,
        warmup_epochs=8,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation pipeline
        augment=True,
        mosaic=0.6,
        mixup=0.3,
        degrees=15.0,
        translate=0.2,
        scale=0.4,
        shear=0.3,
        perspective=0.0005,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        copy_paste=0.4,

        # Detection parameters
        conf=0.15,
        iou=0.55,
        single_cls=False,

        # Loss weights
        box=0.6,
        cls=1.5,
        dfl=1.0,

        # Advanced features
        close_mosaic=20,
        overlap_mask=True,
        cos_lr=True,
    )

    # Execute training with error handling
    try:
        results = model.train(**train_args)
        print(f"\nTraining completed successfully! Results saved in: {run_dir}")

        # Generate training visualization plots
        results_file = os.path.join(run_dir, 'results.json')
        if os.path.exists(results_file):
            plot_training_metrics(results_file, run_dir)
        else:
            print("Warning: Results file not found. Cannot generate training plots.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Cleanup
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
        'epochs': 220,  # Increased to allow full convergence
        'batch_size': 12,  # Optimized for 8GB VRAM
        'img_size': 640,  # Standard YOLO size
        'validate_interval': 2,  # Frequent validation
        'patience': 30  # Extended patience for better convergence
    }

    # Execute training process
    trained_model = train_model(**config)