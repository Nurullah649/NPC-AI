import os
import cv2
import numpy as np
import torch
from .DPVO.dpvo.config import cfg
from .DPVO.dpvo.dpvo import DPVO,Timer


class DPVO_object:
 def __init__(self):
     self.network='/home/nurullah/NPC-AI/Class/DPVO/dpvo.pth'
     self.calib='/home/nurullah/Desktop/Files/calib.txt'
     self.cfg='/home/nurullah/NPC-AI/Class/DPVO/config/npc.yaml'
     self.timeit = False
     self.viz = False
     self.slam=None


 def process_frames_from_list(self,idx,frame):
    """
    Bizim aşağıdaki kodu class halinde tek tek istenen formata çevirip x,y,z line ını geri döndürür
    """
    """
    Belirtilen frame dosya yolları üzerinden frame'leri tek tek okuyarak DPVO işlemini gerçekleştirir.
    Her frame sonrası DPVO'nun pose çıktısından x, y, z koordinatlarını çekip konsola "x,y,z" formatında yazdırır
    ve tüm noktaları 'points.txt' dosyasına kaydeder.
    """

    if os.path.isfile(self.calib):
        intrinsics_default = np.loadtxt(self.calib)
    else:
        intrinsics_default = np.eye(3)

    image = cv2.imread(frame)
    if image is None:
        print(f"Frame okunamadı: {frame}")


    intrinsics = intrinsics_default

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
    if cfg.MIXED_PRECISION:
            image_tensor = image_tensor.half()
    intrinsics_tensor = torch.from_numpy(intrinsics).cuda()

    if self.slam is None:
        _, H, W = image_tensor.shape
        self.slam = DPVO(cfg, self.network, ht=H, wd=W, viz=self.viz)

    with torch.no_grad():
        with Timer("SLAM", enabled=self.timeit):
            self.slam(idx, image_tensor, intrinsics_tensor)

    current_pose = self.slam.get_current_pose()  # 4x4 dönüşüm matrisi
    x = current_pose[0, 3]
    y = current_pose[1, 3]
    z = current_pose[2, 3]

    # Her frame için "x, y, z" formatında çıktı üret ve listeye ekle
    line = f"{x}, {y}, {z}"
    print(idx,"  ",line)
    return x,y,z




"""elapsed_time=time.time()-start_time
        if max_time < elapsed_time:
            max_time=elapsed_time
        print(idx," ",line,f"Time = {elapsed_time:.3f}")
        points_list.append(line)


   del image_tensor, intrinsics_tensor, image
        torch.cuda.empty_cache()
    print(f"Görülen maksimum süre : {max_time}")
    # İşlem tamamlandıktan sonra DPVO'yu kapatıyoruz.
    # Frame işleme döngüsünün sonunda
    if slam is not None:
        # Bellek temizliği yapalım
        torch.cuda.empty_cache()
        # Termination işlemini no_grad bloğu içine alalım
        with torch.no_grad():
            final_trajectory = slam.terminate()
    else:
        final_trajectory = None

    # Noktaları "points.txt" dosyasına kaydedelim.
    with open("points.txt", "w") as f:
        for line in points_list:
            f.write(line + "\n")

    return final_trajectory

if __name__ == '__main__':
    # Sabit ayarlar: Bu değerleri kendi ortamınıza göre düzenleyin.
    network = 'DPVO/dpvo.pth'  # DPVO ağırlık dosyasının yolu
    frame_list_source = '/home/nurullah/Desktop/Files/sim_ana/'
    calib = '/home/nurullah/Desktop/Files/calib.txt'  # Kalibrasyon dosyası yolu
    config_file = 'DPVO/config/npc.yaml'  # DPVO konfigürasyon dosyası

    cfg.merge_from_file(config_file)
    print("Running with config:")
    print(cfg)
    final_trajectory = dpvo(network, calib, cfg)
    frame_list = []
    if os.path.isdir(frame_list_source):
        valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']
        for filename in sorted(os.listdir(frame_list_source)):
            if any(filename.lower().endswith(ext) for ext in valid_exts):
                frame_list.append(os.path.join(frame_list_source, filename))
    else:
        with open(frame_list_source, 'r') as f:
            frame_list = [line.strip() for line in f if line.strip()]

    with open("sonuc.txt", "w") as f:
        for idx, frame_path in enumerate(frame_list):
            f.write(final_trajectory.process_frames_from_list(idx,frame_path)+"\n")"""