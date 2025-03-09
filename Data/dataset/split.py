import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_image_dir = os.path.join('images')  # Merged images
dataset_label_dir = os.path.join('labels')  # Merged labels
split_dir = os.path.join('split')  # Directory to store the split datasets

# Create split directories
train_image_dir = os.path.join(split_dir, 'train', 'images')
train_label_dir = os.path.join(split_dir, 'train', 'labels')
val_image_dir = os.path.join(split_dir, 'val', 'images')
val_label_dir = os.path.join(split_dir, 'val', 'labels')
test_image_dir = os.path.join(split_dir, 'test', 'images')
test_label_dir = os.path.join(split_dir, 'test', 'labels')

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Get list of image files
image_files = os.listdir(dataset_image_dir)
image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.png')]

# Split the dataset (90% train, 5% val, 5% test)
train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)  # 10% for test + val
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)  # Split 10% into 5% val and 5% test

# Function to copy files
def copy_files(files, image_source_dir, label_source_dir, image_dest_dir, label_dest_dir):

    for file_name in files:
        # Copy image
        shutil.copy(os.path.join(image_source_dir, file_name), os.path.join(image_dest_dir, file_name))
        # Copy label
        label_name = os.path.splitext(file_name)[0] + '.txt'
        shutil.copy(os.path.join(label_source_dir, label_name), os.path.join(label_dest_dir, label_name))

# Copy files to respective directories
print("Copying training data...")
copy_files(train_files, dataset_image_dir, dataset_label_dir, train_image_dir, train_label_dir)
print("Copying validation data...")
copy_files(val_files, dataset_image_dir, dataset_label_dir, val_image_dir, val_label_dir)
print("Copying test data...")
copy_files(test_files, dataset_image_dir, dataset_label_dir, test_image_dir, test_label_dir)

print("Dataset splitting complete!")