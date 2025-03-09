import os
import cv2
import numpy as np
import albumentations as A

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

# Function to convert polygons to binary masks
def polygons_to_masks(polygons, image_height, image_width):
    """
    Convert a list of polygons to binary masks (NumPy arrays).
    """
    masks = []
    for polygon in polygons:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
        masks.append(mask)
    return masks

# Function to convert masks back to polygons
def masks_to_polygons(masks):
    """
    Convert binary masks back to polygons.
    """
    polygons = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 3:  # A polygon must have at least 3 points
                polygon = contour.reshape(-1, 2).tolist()
                polygons.append(polygon)
    return polygons

# Define the augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=45, p=0.5),  # Rotate by up to 45 degrees
    A.HorizontalFlip(p=0.5),    # Flip horizontally with 50% probability
    A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Add Gaussian blur
])

# Paths
image_dir = os.path.join('Data', 'labeled_data', 'images')  # Directory containing original images
label_dir = os.path.join('Data', 'labeled_data', 'labels')  # Directory containing YOLO-format polygon labels
output_image_dir = os.path.join('Data', 'augmented_data', 'images')  # Directory to save augmented images
output_label_dir = os.path.join('Data', 'augmented_data', 'labels')  # Directory to save augmented labels

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Number of augmented versions to generate per image
num_augmentations = 3  # Change this to generate more or fewer augmentations

# Loop through all images in the directory
for image_name in os.listdir(image_dir):
    # Load image
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}. Skipping...")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_height, image_width = image.shape[:2]

    # Load corresponding YOLO polygon labels
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {image_name}. Skipping...")
        continue

    # Parse YOLO polygon labels
    polygons, class_ids = parse_yolo_polygon(label_path, image_width, image_height)

    # Convert polygons to binary masks
    masks = polygons_to_masks(polygons, image_height, image_width)

    # Generate multiple augmented versions of the image
    for aug_idx in range(num_augmentations):
        # Apply augmentations
        transformed = transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        # Convert augmented masks back to polygons
        transformed_polygons = masks_to_polygons(transformed_masks)

        # Save augmented image
        augmented_image_name = f'aug_{aug_idx}_{image_name}'
        augmented_image_path = os.path.join(output_image_dir, augmented_image_name)
        cv2.imwrite(augmented_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

        # Save augmented labels
        augmented_label_name = f'aug_{aug_idx}_{os.path.splitext(image_name)[0]}.txt'
        augmented_label_path = os.path.join(output_label_dir, augmented_label_name)
        with open(augmented_label_path, 'w') as f:
            for polygon, class_id in zip(transformed_polygons, class_ids):
                # Normalize polygon coordinates to YOLO format
                normalized_polygon = [(x / transformed_image.shape[1], y / transformed_image.shape[0]) for x, y in polygon]
                yolo_label = [class_id] + [coord for point in normalized_polygon for coord in point]
                f.write(' '.join(map(str, yolo_label)) + '\n')

        print(f"Augmented {image_name} (version {aug_idx + 1}) and saved to {augmented_image_path} and {augmented_label_path}")

print("Data augmentation complete!")