import os
import cv2
import numpy as np
import concurrent.futures
import is_daytime

def split_image(image):
    height, width, _ = image.shape
    part_height = height // 2
    part_width = width // 3
    parts = []
    for i in range(2):
        for j in range(3):
            part = image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            parts.append(part)
    return parts

def process_part(part):
    return is_daytime.is_daytime(part)

def process_image(image_path, desktop_path):
    image = cv2.imread(image_path)
    parts = split_image(image)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_parts = list(executor.map(process_part, parts))

    merged_image = np.vstack([np.hstack(processed_parts[:3]), np.hstack(processed_parts[3:])])
    save_path = os.path.join(desktop_path, "results")
    os.makedirs(save_path, exist_ok=True)
    saved_image_path = os.path.join(save_path, os.path.basename(image_path))
    cv2.imwrite(saved_image_path, merged_image)
    return saved_image_path

