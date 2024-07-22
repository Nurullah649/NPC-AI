import cv2
import os


def video_to_frames(video_path, output_folder="frames", fps=25):
    # Video dosyasını oku
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    # Frame'leri ayırma işlemi
    count = 0
    while success:
        # Frame'yi kaydet
        frame_name = f"{output_folder}/frame_{count:04d}.jpg"
        cv2.imwrite(frame_name, image)

        # Belirtilen fps değerine göre frame'leri atla
        for _ in range(int(vidcap.get(cv2.CAP_PROP_FPS) / fps) - 1):
            vidcap.grab()

        # Sonraki frame'i oku
        success, image = vidcap.read()
        count += 1

    # Video dosyasını kapat
    vidcap.release()


if __name__ == "__main__":
    # Video dosya yolu
    video_path = "/home/nurullah/İndirilenler/video/epm.mp4"
    # Çıktı klasörü
    output_folder = "/home/nurullah/Masaüstü/zenoda-videolar-frame//"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Videoyu frame'lere ayır
    video_to_frames(video_path, output_folder)
