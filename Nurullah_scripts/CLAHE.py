import glob
import os
from os.path import expanduser

import cv2
from matplotlib import pyplot as plt
from colorama import Fore,Style

def clahe(path):
    output_folder = os.path.join(path[:-4], "results")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fname in glob.glob(path):
        img = cv2.imread(fname)
        #print(img.shape)
        # Limunance kanalı için öncelikle bgr to lab formata geçiyoruz.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        # Ham halde dağılımı inceleyelim.
        # plt.hist(l.flat, bins=100, range=(0,255))

        # Geleneksel equalazation yapalım.
        equalized = cv2.equalizeHist(l)

        # Geleneksel equlization sonrası dağılımı inceleyelim.
        # plt.hist(equalized.flat, bins=100, range=(0,255))

        # Kanalları birleştirelim.
        lab_img1_result = cv2.merge((equalized, a, b))

        # Lab formatımızı eski haline bgr haline dönüştürelim
        #hist_eq_img = cv2.cvtColor(lab_img1_result, cv2.COLOR_LAB2BGR)

        # --CLAHE--
        # Aynı işlemleri tekrarlayalım.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Ayrıca 16x16 deneyebiliriz.
        clahe_img = clahe.apply(l)

        # plt.hist(clahe_img.flat, bins=100, range=(0,255))
        lab_img2_result = cv2.merge((clahe_img, a, b))

        CLAHE_result_img = cv2.cvtColor(lab_img2_result, cv2.COLOR_LAB2BGR)

        # Sonuçlar
        #cv2.imwrite(f"CLAHE_DENEME_SONUÇ/img{fname}.jpg", img)
        #cv2.imwrite(f"CLAHE_DENEME_SONUÇ/hist_eq_img.jpg", hist_eq_img)
        filename = os.path.basename(fname)
        cv2.imwrite(os.path.join(output_folder, f"CLAHE_result_img_{filename}"), CLAHE_result_img)
        #print(CLAHE_result_img)
        #print(Fore.RED+f"{fname}"+Style.RESET_ALL)
        #dizi.append(os.path.join(output_folder, f"CLAHE_result_img_{filename}"))
        return os.path.join(output_folder, f"CLAHE_result_img_{filename}")

#for fname in glob.glob(path):
    #print(clahe(path))