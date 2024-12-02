# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory to store all generated graphs
results_dir = "result_graphs"
os.makedirs(results_dir, exist_ok=True)

# Read the training results CSV file
df = pd.read_csv(r'C:\school\ML project files\yoloTestCharm\runs\train7\train\results.csv')

# Set the plotting style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')

# 1. GPU Memory Usage Plot
plt.figure(figsize=(12, 6))
if 'GPU_mem' in df.columns:
    plt.plot(df['epoch'], df['GPU_mem'], label='GPU Memory Usage')
    plt.title('GPU Memory Usage Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('GPU Memory (GB)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'gpu_memory.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Main Training Metrics Plot (2x2 subplot)
fig = plt.figure(figsize=(20, 15))
fig.suptitle('YOLO Training Results Analysis', fontsize=16, y=0.95)

# Plot 1: Loss Values
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
ax1.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
ax1.set_title('Training Losses Over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss Value')
ax1.legend()
ax1.grid(True)

# Plot 2: mAP Values
ax2 = plt.subplot(2, 2, 2)
ax2.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
ax2.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
ax2.set_title('mAP Metrics Over Time')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('mAP Value')
ax2.legend()
ax2.grid(True)

# Plot 3: Precision and Recall
ax3 = plt.subplot(2, 2, 3)
ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
ax3.set_title('Precision and Recall Over Time')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True)

# Plot 4: Learning Rate Schedule
ax4 = plt.subplot(2, 2, 4)
ax4.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2, color='purple')
ax4.set_title('Learning Rate Schedule')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.set_yscale('log')  # Using log scale for better visualization of learning rate decay
ax4.legend()
ax4.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Loss Convergence Analysis
plt.figure(figsize=(12, 6))
window = 10  # Window size for moving average
total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
ma_loss = total_loss.rolling(window=window).mean()
plt.plot(df['epoch'], total_loss, alpha=0.3, label='Total Loss')
plt.plot(df['epoch'], ma_loss, label=f'Moving Average (window={window})')
plt.title('Loss Convergence Analysis')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss_convergence.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Training Progress Heatmap
plt.figure(figsize=(15, 6))
metrics_to_plot = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                   'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']
data_for_heatmap = df[metrics_to_plot].T
plt.imshow(data_for_heatmap, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.yticks(range(len(metrics_to_plot)), [m.split('/')[-1].replace('(B)', '') for m in metrics_to_plot])
plt.xlabel('Epoch')
plt.title('Training Metrics Heatmap')
plt.savefig(os.path.join(results_dir, 'training_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Learning Rate vs Performance Analysis
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()  # Create second y-axis
ax1.plot(df['epoch'], df['lr/pg0'], 'b-', label='Learning Rate')
ax2.plot(df['epoch'], df['metrics/mAP50(B)'], 'r-', label='mAP50')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Learning Rate', color='b')
ax2.set_ylabel('mAP50', color='r')
plt.title('Learning Rate vs mAP50')
ax1.set_yscale('log')
plt.savefig(os.path.join(results_dir, 'lr_vs_map.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Comprehensive Convergence Analysis
plt.figure(figsize=(15, 8))

# Define metrics for convergence analysis
metrics_for_convergence = {
    'Loss': (df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']),
    'mAP50': df['metrics/mAP50(B)'],
    'Precision': df['metrics/precision(B)'],
    'Recall': df['metrics/recall(B)']
}

# Plot normalized convergence for each metric
for metric_name, values in metrics_for_convergence.items():
    if metric_name == 'Loss':
        # Invert loss values so convergence trends upward like other metrics
        normalized = 1 - (values - values.min()) / (values.max() - values.min())
    else:
        normalized = (values - values.min()) / (values.max() - values.min())

    # Calculate and plot moving average for smoother visualization
    ma = normalized.rolling(window=10).mean()
    plt.plot(df['epoch'], ma, label=f'{metric_name} Convergence', linewidth=2)

plt.title('Training Convergence Analysis')
plt.xlabel('Epoch')
plt.ylabel('Normalized Progress (0-1)')
plt.legend()
plt.grid(True)
plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.3, label='90% Convergence')
plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.3, label='95% Convergence')

# Add annotations for convergence points
for metric_name, values in metrics_for_convergence.items():
    normalized = (values - values.min()) / (values.max() - values.min())
    if metric_name == 'Loss':
        normalized = 1 - normalized

    # Find and annotate 90% convergence points
    conv_90 = df['epoch'][normalized >= 0.9].iloc[0] if any(normalized >= 0.9) else None
    if conv_90:
        plt.annotate(f'{metric_name}: 90% at epoch {int(conv_90)}',
                     xy=(conv_90, 0.9),
                     xytext=(10, 10), textcoords='offset points')

plt.savefig(os.path.join(results_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print summary of generated files
print(f"All graphs have been saved in the '{results_dir}' directory:")
print("1. gpu_memory.png - GPU memory usage")
print("2. training_analysis.png - Main training metrics")
print("3. loss_convergence.png - Loss convergence analysis")
print("4. training_heatmap.png - Training progress heatmap")
print("5. lr_vs_map.png - Learning rate vs mAP50")
print("6. convergence_analysis.png - Comprehensive convergence analysis")