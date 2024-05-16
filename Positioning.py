import cv2
import numpy as np




class CameraMovementTracker:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.positions = np.array([0.0, 0.0])  # X ve Y pozisyonlarını saklamak için
        self.prev_rvec = None  # Önceki rotasyon vektörünü saklamak için
        self.prev_tvec = None  # Önceki translasyon vektörünü saklamak için
        self.camera_matrix = None  # Kamera matrisini saklamak için
        self.dist_coeffs = None  # Bozulma katsayılarını saklamak için

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if hasattr(self, 'prev_des'):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.prev_des, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            if len(src_pts) >= 4 and len(dst_pts) >= 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    # Kamera matrisi ve bozulma katsayılarını al
                    if self.camera_matrix is None or self.dist_coeffs is None:
                        self.camera_matrix = np.array([[frame.shape[1], 0, frame.shape[1] / 2],
                                                       [0, frame.shape[0], frame.shape[0] / 2],
                                                       [0, 0, 1]], dtype=np.float64)
                        self.dist_coeffs = np.zeros((4, 1))
                    # Dönüşüm matrisi hesapla
                    _, rvec, tvec, inliers = cv2.solvePnPRansac(np.array([[0, 0, 0]], dtype=np.float32), dst_pts, self.camera_matrix, self.dist_coeffs)
                    # Önceki ve şu anki rotasyon vektörleri arasındaki farkı al
                    if self.prev_rvec is not None:
                        diff_rvec = rvec - self.prev_rvec
                        # Dönme matrisi oluştur
                        R, _ = cv2.Rodrigues(diff_rvec)
                        # Dönme matrisini kullanarak x, y dönüşlerini hesapla
                        angles = np.arctan2(R[1, 0], R[0, 0]), np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
                        # X ve Y pozisyonlarını güncelle, x'i -1 ile çarp ve her ikisini de 40'a böl
                        self.positions += np.array(angles) * np.array([-1, 1]) / 40
                    # Önceki rotasyon vektörünü güncelle
                    self.prev_rvec = rvec
                else:
                    print("Homografi bulunamadı")
            else:
                print("Yeterli eşleşme yok")
        # Önceki karedeki özelliklerini sakla
        self.prev_des = des2
        self.prev_kp = kp2

    def get_positions(self):
        # X ve Y pozisyonlarını döndür
        return self.positions
