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


def get_next_train_number():
    """
    Manages training run numbering by scanning existing directories.
    Ensures each training run has a unique sequential number.

    Returns:
        int: Next available training number starting from 1
    """
    # Define the directory where training runs are saved
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
    Generates a 2x2 grid of plots showing different aspects of the training process.

    Args:
        results_file (str): Path to the JSON file containing training results
        save_dir (str): Directory where the plot will be saved
    """
    # Load the training results from JSON file
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Plot 1: Training Losses
    epochs = list(range(1, len(results['train/box_loss']) + 1))
    ax1.plot(epochs, results['train/box_loss'], label='Box Loss')  # Bounding box regression loss
    ax1.plot(epochs, results['train/cls_loss'], label='Class Loss')  # Classification loss
    ax1.plot(epochs, results['train/dfl_loss'], label='DFL Loss')  # Distribution focal loss
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Validation Metrics
    # Filter out zero values which indicate epochs without validation
    val_epochs = [i for i, j in enumerate(results['metrics/mAP50(B)']) if j != 0]
    val_map50 = [j for j in results['metrics/mAP50(B)'] if j != 0]  # mAP at IoU=0.5
    val_map50_95 = [j for j in results['metrics/mAP50-95(B)'] if j != 0]  # mAP at IoU=0.5:0.95

    ax2.plot(val_epochs, val_map50, label='mAP50')
    ax2.plot(val_epochs, val_map50_95, label='mAP50-95')
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
    ax3.grid(True)

    # Plot 4: Total Loss Convergence
    # Sum all loss components to get total loss
    total_loss = np.array(results['train/box_loss']) + np.array(results['train/cls_loss']) + np.array(
        results['train/dfl_loss'])
    ax4.plot(epochs, total_loss, label='Total Loss')
    ax4.set_title('Loss Convergence')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Total Loss')
    ax4.grid(True)

    # Adjust layout to prevent overlap and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

    print(f"Training metrics plot saved to: {os.path.join(save_dir, 'training_metrics.png')}")


def train_model(yolo_task, mode, data_yaml_path, epochs, batch_size, img_size, validate_interval=5, patience=15):
    """
    Main training function optimized for high recall in object detection.
    Uses pre-calculated weights from data.yaml for class balancing.

    Args:
        data_yaml_path (str): Path to YAML config with dataset info and weights
        epochs (int): Total training epochs
        batch_size (int): Images per batch, optimized for available VRAM
        img_size (int): Input image resolution
        validate_interval (int): Epochs between validation runs
        patience (int): Early stopping counter threshold
    """
    # Create a unique directory for this training run
    train_number = get_next_train_number()
    run_dir = os.path.join("runs", f"train{'' if train_number == 1 else train_number}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created training directory: {run_dir}")

    # Setup GPU/CPU device and optimize GPU settings if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        torch.cuda.empty_cache()  # Clear GPU memory
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN

    # Create temporary YAML file for training configuration
    temp_yaml_path = os.path.join(run_dir, 'temp_data.yaml')
    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Save configuration to temporary file
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)

    # Initialize model with pre-trained weights
    model = YOLO('yolov8s.pt')

    # Configure training parameters
    train_args = dict(
        data=temp_yaml_path,  # Dataset configuration file
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        save=True,  # Enable model checkpointing
        save_period=validate_interval,  # Checkpoint frequency
        patience=patience,  # Early stopping patience
        device=device,
        project=run_dir,
        exist_ok=True,
        pretrained=True,  # Use pre-trained weights
        amp=True,  # Enable automatic mixed precision
        verbose=True,
        cache=False,

        # Optimization parameters for training stability
        lr0=0.0003,  # Initial learning rate
        lrf=0.005,  # Final learning rate factor
        momentum=0.937,  # SGD momentum
        weight_decay=0.001,  # Weight decay coefficient
        warmup_epochs=max(10, int(epochs * 0.15)),  # Warmup period
        warmup_momentum=0.8,  # Initial warmup momentum
        warmup_bias_lr=0.05,  # Warmup bias learning rate

        # Training techniques for better generalization
        label_smoothing=0.1,  # Label smoothing epsilon

        # Data augmentation pipeline for improved robustness
        augment=True,
        mosaic=0.95,  # Mosaic augmentation probability
        mixup=0.5,  # Mixup augmentation probability
        degrees=15.0,  # Random rotation degrees
        translate=0.2,  # Random translation
        scale=0.6,  # Random scaling
        shear=0.3,  # Random shear
        perspective=0.001,  # Random perspective
        flipud=0.3,  # Random flip up-down
        fliplr=0.5,  # Random flip left-right
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        copy_paste=0.5,  # Copy-paste augmentation

        # Detection optimization parameters
        multi_scale=True,  # Enable multi-scale training
        conf=0.15,  # Confidence threshold
        iou=0.5,  # NMS IoU threshold
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
        # Cleanup temporary files
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

    return model


if __name__ == '__main__':
    # Training configuration parameters
    config = {
        'yolo_task': 'detect',
        'mode': 'train',
        'data_yaml_path': r"C:\school\ML project files\yoloTestCharm\data.yaml",
        'epochs': 50,  # Total number of training epochs
        'batch_size': 32,  # Batch size per training iteration
        'img_size': 448,  # Input image resolution
        'validate_interval': 5,  # Epochs between validations
        'patience': 15  # Early stopping patience
    }

    # Execute training process
    trained_model = train_model(**config)