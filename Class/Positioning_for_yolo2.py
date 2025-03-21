import cv2
import numpy as np


class CameraMovementTracker:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.positions = np.array([0.0, 0.0])
        self.current_position = np.array([0.0, 0.0])
        self.current_angle = 0.0
        self.is_first_frame = True
        self.camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
        ])
        self.dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undistorted_frame = cv2.undistort(gray, self.camera_matrix, self.dist_coeffs)
        kp2, des2 = self.orb.detectAndCompute(undistorted_frame, None)
        if hasattr(self, 'prev_des') and self.prev_des is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.prev_des, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            if len(src_pts) >= 4 and len(dst_pts) >= 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    src_mean = np.mean(src_pts, axis=0)
                    dst_mean = np.mean(dst_pts, axis=0)
                    movement = dst_mean - src_mean
                    self.current_position += movement.flatten()

                    # Açısal değişimi hesapla
                    angle_rad = np.arctan2(M[1, 0], M[0, 0])
                    self.current_angle += np.degrees(angle_rad)

                    # Pozisyonları güncelle ve X ve Y eksenlerini ters çevir
                    self.positions = self.current_position.copy() * np.array([-1, -1]) / 45,41671309
                else:
                    self.positions = self.current_position.copy() * np.array([-1, -1]) /45,41671309
            else:
                self.positions = self.current_position.copy() * np.array([-1, -1]) / 45,41671309
        else:
            self.positions = np.array([0.0, 0.0])

        self.prev_des = des2
        self.prev_kp = kp2
        self.is_first_frame = False

    def get_positions(self):
        if self.is_first_frame:
            return np.array([0.0, 0.0])
        return self.positions

    def get_angle(self):
        if self.is_first_frame:
            return 0.0
        return self.current_angle