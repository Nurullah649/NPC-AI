
import cv2
import os

# ORB ve gerekli değişkenlerin tanımlanması
orb = cv2.ORB.create()

# Frame dizini ve sıralama
frames_path = 'downloaded_frames/frames/2024_TUYZ_Online_Yarisma_Ana_Oturum_pmcfrqkz_Video/'
frames = sorted(os.listdir(frames_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
count = 0
for frame in frames:
    frame_path = os.path.join(frames_path, frame)
    frame_name = os.path.basename(frame_path)
    print(f"Processing frame: {frame_name}")
    frame = cv2.imread(frame_path)
    res=orb.detect(frame)
    print(res)

    count += 1
