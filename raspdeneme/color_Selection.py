import numpy as np
import cv2

# Kamera kaynağını başlat (0, varsayılan kamerayı açar)
cap = cv2.VideoCapture(0)

# Kırmızı için alt ve üst sınırlar (HSV formatında)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Genişletilmiş mavi renk aralıkları (HSV formatında)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([150, 255, 255])

# Kameranın çözünürlüğünü öğren ve merkezini bul
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_center = (frame_width // 2, frame_height // 2)

# Nesne takipçisi (CSRT Tracker) kullanacağız
tracker = cv2.TrackerCSRT.create()
tracking_mode = False  # Takip modunun aktif olup olmadığını belirler
bbox = None  # Takip edilen nesnenin bounding box'ı

while True:
    # Kameradan bir kare yakala
    ret, frame = cap.read()

    # Kameradan görüntü alınıp alınmadığını kontrol et
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # HSV renk uzayına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not tracking_mode:
        # Kırmızı rengi tespit et (iki farklı aralığı birleştir)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2

        # Mavi rengi tespit et (genişletilmiş aralık)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Kırmızı ve mavi için konturları bul
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # En yakın nesnenin merkezini ve bbox bilgilerini sakla
        min_distance = float('inf')
        closest_bbox = None
        closest_center = None
        closest_color = None  # En yakın nesnenin rengini saklamak için

        # Kırmızı nesneleri kontrol et
        for contour in contours_red:
            if cv2.contourArea(contour) > 500:  # Küçük nesneleri göz ardı et
                x, y, w, h = cv2.boundingRect(contour)
                red_center = (x + w // 2, y + h // 2)

                # Kameranın merkezi ile nesnenin merkezi arasındaki mesafeyi hesapla
                distance = np.sqrt((red_center[0] - frame_center[0])**2 + (red_center[1] - frame_center[1])**2)

                # Daha yakın bir nesne varsa kaydet
                if distance < min_distance:
                    min_distance = distance
                    closest_bbox = (x, y, w, h)
                    closest_center = red_center
                    closest_color = "Red"  # Renk kırmızı

        # Mavi nesneleri kontrol et
        for contour in contours_blue:
            if cv2.contourArea(contour) > 500:  # Küçük nesneleri göz ardı et
                x, y, w, h = cv2.boundingRect(contour)
                blue_center = (x + w // 2, y + h // 2)

                # Kameranın merkezi ile nesnenin merkezi arasındaki mesafeyi hesapla
                distance = np.sqrt((blue_center[0] - frame_center[0])**2 + (blue_center[1] - frame_center[1])**2)

                # Daha yakın bir nesne varsa kaydet
                if distance < min_distance:
                    min_distance = distance
                    closest_bbox = (x, y, w, h)
                    closest_center = blue_center
                    closest_color = "Blue"  # Renk mavi

        # En yakın nesneyi bulduysak takip moduna geç
        if closest_bbox is not None:
            bbox = closest_bbox
            tracking_mode = True
            tracker.init(frame, bbox)  # CSRT tracker'ı başlat
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Closest: {closest_color}, X={closest_center[0] - frame_center[0]}, Y={closest_center[1] - frame_center[1]}",
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        # Takip modundaysa, nesneyi takip et
        success, bbox = tracker.update(frame)

        if success:
            # Takip edilen nesnenin bbox'ını güncelle ve ekrana çiz
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

            # Nesnenin kameranın merkezine olan uzaklığını hesapla
            object_center = (int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2))
            cv2.putText(frame, f"Tracking: X={object_center[0] - frame_center[0]}, Y={object_center[1] - frame_center[1]}",
                        (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Takip kaybedilirse
            cv2.putText(frame, "Lost Tracking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tracking_mode = False

    # Kameranın merkezini ekranda göster
    cv2.circle(frame, frame_center, 5, (0, 255, 255), -1)

    # Orijinal görüntüyü göster
    cv2.imshow('Red and Blue Object Detection and Tracking', frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
