import os
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import leastsq, least_squares
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter
#from Class.R_Model import R_Model


def train_model(alg_postions, gt_positions, scale_factor, offset):
    with open("../data/ALG_data_model_train.csv", 'a') as file:
        file.write("x,y\n")
        for position in alg_postions:
            scaled_positions = np.dot(position, scale_factor.T) + offset
            pred_translation_x = scaled_positions[0]
            pred_translation_y = scaled_positions[1]
            file.write(f"{pred_translation_x},{pred_translation_y}\n")

    with open("../data/GT_data_model_train.csv", 'a') as file:
        file.write("x,y\n")
        for position in gt_positions:
            file.write(f"{position[0]},{position[1]}\n")
    # r_model = R_Model(alg_data_path="R_model_train.csv",gt_data_path="GT_model_train.csv")
    # return r_model


def initialize_orb():
    return cv2.ORB.create(nfeatures=1050, WTA_K=3, scaleFactor=1.035, edgeThreshold=14, nlevels=9)


def initialize_kalman_filter(initial_position):
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


def kalman_update(kf, measured_position):
    kf.predict()
    kf.update(measured_position)
    return kf.x[0], kf.x[2]


def match_features(prev_des, des2, prev_kp, kp2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(prev_des, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts


def residuals(h, src_pts, dst_pts):
    h_matrix = h.reshape((3, 3))
    projected_pts = cv2.perspectiveTransform(src_pts, h_matrix)
    return (dst_pts - projected_pts).flatten()


def optimize_homography(m, src_pts, dst_pts):
    initial_h = m.flatten()
    optimized_h, _ = leastsq(residuals, initial_h, args=(src_pts, dst_pts))
    optimized_h_matrix = optimized_h.reshape((3, 3))
    return optimized_h_matrix


def update_position(src_pts, dst_pts, current_position, current_angle):
    positions = current_position.copy()
    if len(src_pts) >= 4 and len(dst_pts) >= 4:

        m, mask = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC,ransacReprojThreshold=3.0)
        if m is not None:
            m = optimize_homography(m, src_pts, dst_pts)  # LM ile optimizasyon
            src_mean = np.mean(src_pts, axis=0)
            dst_mean = np.mean(dst_pts, axis=0)
            movement = dst_mean - src_mean
            angle_rad = np.arctan2(m[1, 0], m[0, 0])
            if np.abs(np.degrees(angle_rad)) > 1:
                current_angle += np.degrees(angle_rad)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            movement_rotated = np.dot(rotation_matrix, movement.flatten())
            positions += movement_rotated
    return positions, current_angle


def calculate_scaling_factors(alg_positions, xy_data):
    model = LinearRegression(fit_intercept=False, positive=False)
    model.fit(alg_positions, xy_data)
    scale_factor = model.coef_
    offset = model.intercept_
    return scale_factor, offset


def process_frame(frame, count, frame_name, xy_data, orb, kf, prev_des, prev_kp, current_position, current_angle,
                  alg_positions, scale_factor, offset):
    camera_matrix = np.array([
        [1.4133e+03, 0, 950.0639],
        [0, 1.4188e+03, 543.3796],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([-0.0091, 0.0666, 0, 0])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted_frame = cv2.undistort(gray, camera_matrix, dist_coeffs)
    kp2, des2 = orb.detectAndCompute(undistorted_frame, None)

    if prev_des is not None:
        src_pts, dst_pts = match_features(prev_des, des2, prev_kp, kp2)
        current_position, current_angle = update_position(src_pts, dst_pts, current_position, current_angle)

    else:
        current_position = np.array([xy_data[0][0], xy_data[0][1]])

    prev_des = des2
    prev_kp = kp2
    alg_positions.append(current_position)

    #kalman_position = kalman_update(kf, current_position)

    if count >= 449 and scale_factor is None:
        scale_factor, offset = calculate_scaling_factors(alg_positions, xy_data)
        #train_model(alg_positions, xy_data, scale_factor, offset)

    if scale_factor is not None:
        scaled_positions = np.dot(current_position, scale_factor.T) + offset
        pred_translation_x = scaled_positions[0]
        pred_translation_y = scaled_positions[1]
        # pred_translation_x,pred_translation_y=r_model.predict(pred_translation_x,pred_translation_y)
    else:
        pred_translation_x = xy_data[count][0]
        pred_translation_y = xy_data[count][1]

    with open("Result_2.txt", 'a') as file:
        file.write(f"{pred_translation_x}, {pred_translation_y}\n")

    return prev_des, prev_kp, current_position, current_angle, scale_factor, offset


def main():
    orb = initialize_orb()
    file_path = '../../Predict/2024_TUYZ_Online_Yarisma_Oturumu/2024_TUYZ_Online_Yarisma_Ana_Oturum.csv'
    df = pd.read_csv(file_path)
    x_sutunu_indeksi = 0
    y_sutunu_indeksi = 1
    xy_data = df.iloc[:450, [x_sutunu_indeksi, y_sutunu_indeksi]].values

    positions = np.array([0.0, 0.0])
    current_position = np.array([xy_data[0][0], xy_data[0][1]])
    current_angle = 0.0
    is_first_frame = True
    prev_des = None
    prev_kp = None
    scale_factor = None
    offset = None

    frames_path = '../../Predict/2024_TUYZ_Online_Yarisma_Oturumu/2024_TUYZ_Online_Yarisma_Ana_Oturum/'
    frames = sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
    alg_positions = []

    kf = initialize_kalman_filter(current_position)

    count = 0
    for frame in frames:
        frame_path = os.path.join(frames_path, frame)
        frame_name = os.path.basename(frame_path)
        print(f"Processing frame: {frame_name}")
        frame = cv2.imread(frame_path)
        prev_des, prev_kp, current_position, current_angle, scale_factor, offset = process_frame(
            frame, count, frame_name, xy_data, orb, kf, prev_des, prev_kp, current_position,
            current_angle, alg_positions, scale_factor, offset)
        count += 1


if __name__ == "__main__":
    main()
