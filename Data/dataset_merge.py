import os
import shutil


original_image_dir = os.path.join('labeled_data', 'images')
original_label_dir = os.path.join('labeled_data', 'labels')
augmented_image_dir = os.path.join('augmented_data', 'images')
augmented_label_dir = os.path.join('augmented_data', 'labels')
dataset_image_dir = os.path.join('dataset', 'images')
dataset_label_dir = os.path.join('dataset', 'labels')


os.makedirs(dataset_image_dir, exist_ok=True)
os.makedirs(dataset_label_dir, exist_ok=True)


def copy_files(source_dir, destination_dir):

    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copy(source_path, destination_path)
        print(f"Copied {file_name} to {destination_dir}")


copy_files(original_image_dir, dataset_image_dir)
copy_files(original_label_dir, dataset_label_dir)

copy_files(augmented_image_dir, dataset_image_dir)
copy_files(augmented_label_dir, dataset_label_dir)

print("Done")