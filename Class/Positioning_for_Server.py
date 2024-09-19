import cv2
import numpy as np
from scipy.optimize import leastsq
from filterpy.kalman import KalmanFilter
camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
    ])
dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])
class CameraMovementTracker:
    def __init__(self):
        self.orb = cv2.ORB.create(nfeatures=1050, WTA_K=3, scaleFactor=1.035, edgeThreshold=14, nlevels=9)
        self.positions = np.array([0.0, 0.0])
        self.current_position = np.array([0.0, 0.0])
        self.current_angle = 0.0
        self.is_first_frame = True
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        #self.first_frame=first_frame
        #self.kf=self.initialize_kalman_filter(self.first_frame)

    def initialize_kalman_filter(self,initial_position):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([initial_position[0], 0., initial_position[1], 0.])
        kf.F = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        kf.P *= 50000
        kf.R = np.array([
            [50000, 0],
            [0, 50000]
        ])
        kf.Q = np.eye(4)
        return kf

    def kalman_update(self,kf, measured_position):
        kf.predict()
        kf.update(measured_position)
        return kf.x[0], kf.x[2]


    def residuals(self,h, src_pts, dst_pts):
        h_matrix = h.reshape((3, 3))
        projected_pts = cv2.perspectiveTransform(src_pts, h_matrix)
        return (dst_pts - projected_pts).flatten()

    def optimize_homography(self,m, src_pts, dst_pts):
        initial_h = m.flatten()
        optimized_h, _ = leastsq(self.residuals, initial_h, args=(src_pts, dst_pts))
        optimized_h_matrix = optimized_h.reshape((3, 3))
        return optimized_h_matrix


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
                m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if m is not None:
                    m = self.optimize_homography(m, src_pts, dst_pts)
                    src_mean = np.mean(src_pts, axis=0)
                    dst_mean = np.mean(dst_pts, axis=0)
                    movement = dst_mean - src_mean
                    angle_rad = np.arctan2(m[1, 0], m[0, 0])
                    if np.abs(np.degrees(angle_rad)) > 1:
                        self.current_angle += np.degrees(angle_rad)
                    rotation_matrix = np.array([
                        [np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]
                    ])
                    movement_rotated = np.dot(rotation_matrix, movement.flatten())
                    self.positions = self.positions+movement_rotated
                    #self.positions = self.kalman_update(self.kf, self.positions)
                else:
                    self.positions = self.current_position.copy() / 40
            else:
                self.positions = self.current_position.copy() / 40
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
