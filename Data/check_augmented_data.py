import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to parse YOLO polygon labels
def parse_yolo_polygon(label_path, image_width, image_height):
    """
    Parse YOLO polygon labels and return polygons and class IDs.
    Each line in the label file is: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    """
    polygons = []
    class_ids = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            polygon = [(x * image_width, y * image_height) for x, y in zip(parts[1::2], parts[2::2])]
            polygons.append(polygon)
            class_ids.append(class_id)
    return polygons, class_ids

# Function to visualize augmented image and labels
def visualize_augmented_image_and_labels(image_path, label_path):
    """
    Visualize an augmented image and its corresponding YOLO polygon labels.
    """
    # Load augmented image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_height, image_width = image.shape[:2]

    # Load and parse augmented YOLO polygon labels
    if not os.path.exists(label_path):
        print(f"Error: No label file found for {label_path}.")
        return
    polygons, class_ids = parse_yolo_polygon(label_path, image_width, image_height)

    # Plot the augmented image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Plot the augmented polygons
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32)
        plt.plot(*zip(*polygon), color='red', linewidth=2, marker='o', markersize=5)
        plt.fill(*zip(*polygon), color='red', alpha=0.3)  # Fill the polygon

    # Add title and show the plot
    plt.title(f"Augmented Image: {os.path.basename(image_path)}\nLabels: {len(polygons)} polygons")
    plt.axis('off')
    plt.show()

# Paths to augmented data
augmented_image_dir = os.path.join('Data', 'augmented_data', 'images')  # Directory containing augmented images
augmented_label_dir = os.path.join('Data', 'augmented_data', 'labels')  # Directory containing augmented labels

# Loop through augmented images and visualize them
for image_name in os.listdir(augmented_image_dir):
    image_path = os.path.join(augmented_image_dir, image_name)
    label_path = os.path.join(augmented_label_dir, os.path.splitext(image_name)[0] + '.txt')
    
    # Visualize the augmented image and labels
    visualize_augmented_image_and_labels(image_path, label_path)