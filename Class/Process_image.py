import os
import cv2
import numpy as np
import concurrent.futures
import Class.is_daytime

def process_image(image_path,destkop_path):
    # Resmi yükleyin
    image = cv2.imread(image_path)
    # Resmi 2x3 (6 parça) parçaya bölme
    height, width, _ = image.shape
    part_height = height // 2
    part_width = width // 3
    parts = []
    for i in range(2):
        for j in range(3):
            part = image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            parts.append(part)

    # Her parçayı ön işlemeye sokma
    def process_part(part):
        return Class.is_daytime.is_daytime(part)

    # Parçaları paralel şekilde ön işleme sokma
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_parts = list(executor.map(process_part, parts))
    merged_image = np.vstack([np.hstack(processed_parts[:3]), np.hstack(processed_parts[3:])])
    save_path=f"{destkop_path}/results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saved_image_path=f"{destkop_path}/results/{os.path.basename(image_path)}"
    cv2.imwrite(saved_image_path, merged_image)
    return saved_image_path
