import os
import time  # Import time module for measuring process durations
import onnxruntime as ort
import numpy as np
import cv2

# 1. Load the ONNX model
onnx_model_path = '../runs/detect/yolov10x-1920/best.onnx'
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model file not found at {onnx_model_path}")

# Measure model loading time
start_time = time.time()
session_options = ort.SessionOptions()
session_options.log_severity_level = 1
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.4f} seconds")

# 2. Get input name and shape expected by the model
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# 3. Load and preprocess the input image directory
img_path = '/home/nurullah/Desktop/Predict/2024_TUYZ_Online_Yarisma_Oturumu/2024_TUYZ_Online_Yarisma_Ana_Oturum/'
frames = sorted(os.listdir(img_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

# Loop through each image frame in the directory
for img in frames:
    image_path = os.path.join(img_path, img)
    print(f"Processing: {image_path}")

    # Measure image loading time
    start_time = time.time()
    image = cv2.imread(image_path)
    image_load_time = time.time() - start_time
    print(f"Image loaded in {image_load_time:.4f} seconds")

    if image is None:
        print(f"Warning: Unable to load image {image_path}. Skipping.")
        continue

    original_image = image.copy()
    h, w = input_shape[2], input_shape[3]  # Expected height and width by the model
    print(f"Expected input size: {h}x{w}")

    # Measure preprocessing time
    start_time = time.time()
    image_resized = cv2.resize(image, (w, h))
    image_data = image_resized.astype(np.float32)
    image_data /= 255.0  # Normalize pixel values to [0, 1]
    image_data = np.transpose(image_data, (2, 0, 1))  # Convert from HWC to CHW format
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    preprocessing_time = time.time() - start_time
    print(f"Image preprocessing done in {preprocessing_time:.4f} seconds")

    # 4. Run inference
    start_time = time.time()
    output = session.run(None, {input_name: image_data})
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")

    # 5. Post-process the output
    start_time = time.time()
    detections = output[0]

    # Process object detections
    boxes = []
    for detection in detections[0]:
        x_center, y_center, width, height, confidence = detection[:5]
        if confidence > 0.5:  # Filter detections above a certain confidence threshold
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            boxes.append([x1, y1, x2, y2])
    postprocessing_time = time.time() - start_time
    print(f"Post-processing done in {postprocessing_time:.4f} seconds")

    # 6. Visualize the results
    start_time = time.time()
    for box in boxes:
        print(f"Detected object: {box[0]}, {box[1]}, {box[2]}, {box[3]}")
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow('Detected Image', original_image)
    visualization_time = time.time() - start_time
    print(f"Visualization done in {visualization_time:.4f} seconds")

    # Wait 1 millisecond between frames or until 'q' is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows once processing is complete
cv2.destroyAllWindows()
