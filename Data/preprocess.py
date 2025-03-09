import os
import cv2
import numpy as np


input = 'Data/images/'
output = "Data/preprocessed_images/"

os.makedirs(output, exist_ok=True)

img_size = (640, 640)

def resize_with_padding(image, target_size):
    h, w, _ = image.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while keeping aspect ratio
    resized_img = cv2.resize(image, (new_w, new_h))

    # Create a blank (black) image
    padded_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Center the resized image
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    return padded_img

for img_name in os.listdir(input):

    img_path = os.path.join(input, img_name)
    img = cv2.imread(img_path)

    if img is not None : 

        # img_resized = cv2.resize(img, img_size)
        img_resized = resize_with_padding(img, img_size)

        output_path = os.path.join(output, img_name)
        cv2.imwrite(output_path, img_resized)


