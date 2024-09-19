import onnxruntime as ort
import numpy as np
import cv2

# 1. ONNX modelini yükleyin
onnx_model_path = 'yolo_model.onnx'
session = ort.InferenceSession(onnx_model_path)

# 2. Modelin beklediği girişlerin adını alın
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# 3. Girdi görüntüsünü yükleyin ve ön işleme yapın
image_path = '../downloaded_frames/frames/deneme/frame_007620.jpg'
image = cv2.imread(image_path)
original_image = image.copy()
h, w = input_shape[2], input_shape[3]  # Modelin beklediği yükseklik ve genişlik
image_resized = cv2.resize(image, (w, h))
image_data = image_resized.astype(np.float32)
image_data /= 255.0  # Normalizasyon
image_data = np.transpose(image_data, (2, 0, 1))  # HWC to CHW format
image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

# 4. Modelden tahmin alın
output = session.run(None, {input_name: image_data})

# 5. Çıktıyı işleme
# Bu adım, YOLO'nun çıkış formatına bağlıdır. Genelde, bir çıkış tensörü alırsınız ve
# bu tensörden nesne sınıfları, konumlar ve güven skorlarını çıkarmak için ek işleme yapmanız gerekir.
detections = output[0]

# Örnek olarak, nesne algılamalarını işlemeye devam edelim
# detection: [batch, num_boxes, 5 + num_classes] formatında olabilir
# 5 + num_classes: [x_center, y_center, width, height, confidence, ...class_scores]
boxes = []
for detection in detections[0]:
    x_center, y_center, width, height, confidence = detection[:5]
    if confidence > 0.5:  # Belirli bir eşik değeri üzerinde olanları filtreleyin
        x1 = int((x_center - width / 2) * original_image.shape[1])
        y1 = int((y_center - height / 2) * original_image.shape[0])
        x2 = int((x_center + width / 2) * original_image.shape[1])
        y2 = int((y_center + height / 2) * original_image.shape[0])
        boxes.append([x1, y1, x2, y2])

# 6. Sonuçları görselleştirin
for box in boxes:
    cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

cv2.imshow('Detected Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
