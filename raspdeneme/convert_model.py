import torch
from yolov10.ultralytics import YOLOv10
model=YOLOv10('/home/nurullah/Desktop/NPC-AI/runs/detect/yolov10x-1920/best.pt')
print(model)
dummy_input = torch.randn(1, 3, 1088, 1920)

# 2. Modeli ONNX formatına dönüştürün
onnx_model_path = "yolov10_model.onnx"
torch.onnx.export(
    model,               # Model nesnesi
    dummy_input,         # Girdi tensorü
    onnx_model_path,     # Kaydedilecek ONNX dosyasının yolu
    export_params=True,  # Model parametrelerini de kaydedin
    opset_version=11,    # ONNX opset versiyonu
    do_constant_folding=True,  # Sabit katmanları katlayın
    input_names=['input'],   # Girdi adı
    output_names=['output'],  # Çıktı adı
    dynamic_axes={'input': {0: 'batch_size'},    # Dinamik boyutlar (isteğe bağlı)
                  'output': {0: 'batch_size'}}
)

print(f"Model başarıyla '{onnx_model_path}' olarak kaydedildi.")