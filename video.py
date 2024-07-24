import cv2
import os


def images_to_video(image_folder, video_name, fps=30):
    # List all files in the directory
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure the images are in the correct order

    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f'Video created successfully: {video_name}')


# Kullanım
image_folder = 'yolov10/runs/detect/predict34/'  # Görsellerin olduğu dizin
video_name = 'output_video_n.avi'  # Oluşturulacak video dosyasının adı
fps = 7.5  # Videonun FPS değeri

images_to_video(image_folder, video_name, fps)
