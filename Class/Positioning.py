<<<<<<< Updated upstream:Class/Positioning.py
import os
import time
=======
>>>>>>> Stashed changes:Positioning.py
import cv2
import numpy as np
from orbslam3 import ORB_SLAM3  # Bu, ORB-SLAM3 Python binding'i varsayılarak eklenmiştir. Gerçek kullanıma bağlı olarak yüklenmesi gerekebilir.
import sys
import os


def initialize_slam(vocab_path, config_path):
    # ORB-SLAM3 sistemini başlat
    slam = orbslam3.System(config_path, vocab_path, orbslam3.Sensor.MONOCULAR)
    slam.initialize()
    return slam


def process_video(slam, video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Sonuçları kaydetmek için liste oluştur
    poses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Kareyi gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ORB-SLAM3'e kareyi ver
        pose = slam.process_image_mono(gray, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        # Pozisyonu kaydet
        if pose is not None:
            poses.append(pose)
            print("Camera Pose:", pose)

    cap.release()
    cv2.destroyAllWindows()

    # Pozisyonları kaydet
    save_poses(output_path, poses)
def save_poses(output_path, poses):
    with open(output_path, 'w') as f:
        for pose in poses:
            pose_str = ' '.join(map(str, pose.flatten()))
            f.write(pose_str + '\n')


<<<<<<< Updated upstream:Class/Positioning.py

=======
import matplotlib.pyplot as plt


def plot_trajectory(output_path):
    poses = np.loadtxt(output_path)

    # Pozisyonları X, Y, Z olarak ayır
    x = poses[:, 3]
    y = poses[:, 7]
    z = poses[:, 11]

    # Trajektoriyi çiz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Camera Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python script.py path_to_vocabulary path_to_settings path_to_video output_path")
        sys.exit()

    vocab_path = sys.argv[1]
    config_path = sys.argv[2]
    video_path = sys.argv[3]
    output_path = sys.argv[4]

    # ORB-SLAM3'ü başlat
    slam = initialize_slam(vocab_path, config_path)

    # Videoyu işle ve pozisyonları kaydet
    process_video(slam, video_path, output_path)

    # ORB-SLAM3'ü kapat
    slam.shutdown()

    # Trajektoriyi görselleştir
    plot_trajectory(output_path)
>>>>>>> Stashed changes:Positioning.py
