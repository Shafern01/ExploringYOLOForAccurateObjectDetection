import os
from collections import Counter
import numpy as np
import yaml

"""
This script calculates class weights based on annotation frequencies, 
associates the weights with class names, and updates the data.yaml file. 
Class weights help balance imbalanced datasets during training.
"""

def calculate_class_weights(label_dir, num_classes, class_names):
    """
    Calculates class weights based on the frequency of annotations and associates them with class names.
    Args:
        label_dir (str): Directory containing YOLO-format label files.
        num_classes (int): Total number of classes in the dataset.
        class_names (list): List of class names corresponding to class IDs.
    Returns:
        dict: Dictionary mapping class names to normalized weights.
    """
    class_counts = Counter()

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])  # Extract class ID
                    class_counts[class_id] += 1

    total = sum(class_counts.values())

    if total == 0:
        raise ValueError("No annotations found in the label directory.")

    # Calculate weights (inverse frequency)
    weights = [total / class_counts.get(i, 1) for i in range(num_classes)]

    # Normalize weights to sum to 1
    normalized_weights = np.array(weights) / sum(weights)

    # Map weights to class names
    class_weights = {class_names[i]: normalized_weights[i] for i in range(num_classes)}

    # Print class distribution for reference
    print("Class Distribution and Weights:")
    for class_id, count in class_counts.items():
        print(f"Class {class_names[class_id]}: {count} instances, Weight: {class_weights[class_names[class_id]]:.6f}")

    return class_weights


def update_data_yaml(data_yaml_path, class_weights):
    """
    Updates the data.yaml file with the calculated class weights.
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        class_weights (dict): Dictionary of class names and their weights.
    """
    with open(data_yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Add or overwrite the weights key
    data["weights"] = {name: float(weight) for name, weight in class_weights.items()}

    with open(data_yaml_path, "w") as file:
        yaml.safe_dump(data, file)
    print(f"Updated {data_yaml_path} with class weights.")


if __name__ == "__main__":
    # Paths
    label_dir = "C:/school/ML project files/yoloTestCharm/outputValAnnotation"  # Directory with .txt label files
    data_yaml_path = "C:/school/ML project files/yoloTestCharm/data.yaml"  # Path to your data.yaml file

    # Load class names from data.yaml
    with open(data_yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
        class_names = data_yaml["names"]
        num_classes = len(class_names)

    # Calculate and update class weights
    class_weights = calculate_class_weights(label_dir, num_classes, class_names)
    print("Calculated Class Weights:", class_weights)
    update_data_yaml(data_yaml_path, class_weights)
