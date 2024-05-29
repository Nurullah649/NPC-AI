import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('Content/Su_birikintisi.mp4')
orb = cv2.ORB_create()  # orbSlam oluşturma
positions = []  # hareket listesi
ret, prev_frame = cap.read()  # kare oku
if not ret:
    print("Videoyu okuyamadı")
    cap.release()
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # kareyi gri ye çevir
kp1, des1 = orb.detectAndCompute(prev_gray, None)  # anahtar noktaları bul
current_position = np.array([0.0, 0.0])  # başlangıç posizyonu

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)  # sonraki kare
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # anahtar noktaları eşle
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)  # eşleşmeyi sırala
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)#kaynak noktaları
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)# hedef noktaları numpy dizisine dönüştürmek
        if len(src_pts) >= 4 and len(dst_pts) >= 4:  # Homografi için en az 4 noktaya ihtiyaç var
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5,0)
            if M is not None:  # kamera hareketini hesapla ve biiktir
                src_mean = np.mean(src_pts, axis=0)#kaynak noktalarını x vwe y ye göre numpy dizisine aktar
                dst_mean = np.mean(dst_pts, axis=0)#hedef noktalarıda aynı şekilde
                movement = dst_mean - src_mean#hedeften kaynak noktaları çıkarılı ve hareket bulunr
                current_position += movement.flatten()#harekete eklenir
                positions.append(current_position.copy())#hareket ilk posziyona eklenir
            else:
                positions.append(current_position.copy())#M matrsi boş ise tekrar kopyalanır
        else:
            positions.append(current_position.copy())#eğer yeterince nokta alınmamışşa homografi matrisi için kopyalanır
    else:
        positions.append(current_position.copy())#des1 veya des2 değişkenlerininin boş olması durumunda hesaplama olmayacağından tekrar kopyalanır

    # Bir sonraki adım için güncelle
    prev_gray = gray
    kp1, des1 = kp2, des2
cap.release()
# GRAFİK
positions = np.array(positions)
plt.figure(figsize=(10, 6))
plt.plot(-positions[:, 0]/40*110/25, positions[:, 1]/40*110/25, marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kamera hareketine göre 2d grafik ')
plt.grid()
plt.show()