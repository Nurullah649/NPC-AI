import pyrealsense2 as rs
import numpy as np
import cv2

# RealSense akış ayarları
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Pipeline başlat
pipeline.start(config)

try:
    while True:
        # Frame alın
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Görüntüleri Numpy dizilerine dönüştür
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Derinlik görüntüsünü normalleştir ve 8 bitlik formata dönüştür
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_8bit = np.uint8(depth_image_normalized)  # 8 bitlik formata dönüştür

        # Normalleştirilmiş derinlik görüntüsüne renk haritası uygula
        depth_image_colored = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

        # Görüntüleri ekrana bas
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image_colored)

        key = cv2.waitKey(1)
        if key == 27:  # 'ESC' tuşuna basıldığında döngüden çık
            break

finally:
    # Pipeline'ı durdur
    pipeline.stop()
    cv2.destroyAllWindows()
