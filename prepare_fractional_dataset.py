import os
import random


def verify_file_correspondence(images_dir, labels_dir):
    """
    Checks for file correspondence between images and labels directories.

    Args:
        images_dir (str): Path to the directory containing image files.
        labels_dir (str): Path to the directory containing label files.

    Returns:
        dict: A dictionary with lists of missing images and missing labels.
    """
    # Get sets of files without extensions
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg')}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

    # Find discrepancies
    missing_images = label_files - image_files  # Labels without corresponding images
    missing_labels = image_files - label_files  # Images without corresponding labels

    return {
        "missing_images": list(missing_images),
        "missing_labels": list(missing_labels),
    }


def create_fractional_dataset(image_paths, label_paths, fraction):
    """
    Create a fractional dataset by sampling a fraction of the given image and label paths.

    Args:
        image_paths (list): List of image file paths.
        label_paths (list): List of corresponding label file paths.
        fraction (float): Fraction of the dataset to sample (e.g., 0.8 for 80%).

    Returns:
        list: Fractional dataset of image-label pairs.
    """
    total_files = len(image_paths)
    sampled_size = int(total_files * fraction)
    sampled_indices = random.sample(range(total_files), sampled_size)

    fractional_dataset = [(image_paths[i], label_paths[i]) for i in sampled_indices]
    return fractional_dataset


# Directories
images_dir = r"C:\Users\natha\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1\train\images"
labels_dir = r"C:\Users\natha\.cache\kagglehub\datasets\marquis03\bdd100k\versions\1\train\labels"

# Verify correspondence
verification_results = verify_file_correspondence(images_dir, labels_dir)

# Save correspondence results to a file
correspondence_file = "file_correspondence_results.txt"
with open(correspondence_file, "w") as f:
    f.write("Missing Images (labels without images):\n")
    f.write("\n".join(verification_results["missing_images"]) + "\n\n")
    f.write("Missing Labels (images without labels):\n")
    f.write("\n".join(verification_results["missing_labels"]))

print(f"File correspondence check completed. Results saved to {correspondence_file}")

# Filter out images without labels
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg')}
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')}

valid_files = image_files.intersection(label_files)  # Only files that exist in both directories
image_paths = [os.path.join(images_dir, f"{file}.jpg") for file in valid_files]
label_paths = [os.path.join(labels_dir, f"{file}.txt") for file in valid_files]

print(f"Valid image-label pairs: {len(image_paths)}")

# Prepare the fractional dataset
fraction = 0.8  # Fraction to sample
fractional_dataset = create_fractional_dataset(image_paths, label_paths, fraction)

# Save fractional dataset paths to files
output_image_file = "fractional_images.txt"
output_label_file = "fractional_labels.txt"

with open(output_image_file, "w") as f:
    f.write("\n".join([pair[0] for pair in fractional_dataset]))
with open(output_label_file, "w") as f:
    f.write("\n".join([pair[1] for pair in fractional_dataset]))

print(
    f"Fractional dataset created. Image paths saved to {output_image_file}, label paths saved to {output_label_file}.")
