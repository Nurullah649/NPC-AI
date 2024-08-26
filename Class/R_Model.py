# We will define a class R_Model which will be used to train and predict using R models
import os
# Set R environment variables
os.environ['R_HOME'] = r'/usr/lib/R'
os.environ['PATH'] = r'/usr/lib/R/bin:' + os.environ['PATH']
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from sklearn.metrics import mean_absolute_error



class R_Model:

    def train_model(self):

        #r kodunu tetikler ve eğitim yapar
        r_script_path = "/home/nurullah/new_positioning_model/Positioning.R"
        # R script dosyasını çalıştırın
        ro.r['source'](r_script_path)


    def predict(self,x,y):

        #egitilmis R modeli ile tahmin yapar
        # Gerekli R kütüphanelerini yükle
        neuralnet = importr('neuralnet')

        # Pandas DataFrame'i R data frame'e dönüştürme ayarı
        pandas2ri.activate()

        # R modellerini yükle
        ro.r('load("/home/nurullah/new_positioning_model/neuralnet_model_x_new.RData")')
        ro.r('load("/home/nurullah/new_positioning_model/neuralnet_model_y_new.RData")')

        # Ters Z-score için orijinal veri setinden ortalama ve standart sapma değerlerini hesapla
        egitim_orjinal = pd.read_csv("/home/nurullah/new_positioning_model/egitim_orjinal.csv")
        egitim_mean_x = egitim_orjinal['translation_x'].mean()
        egitim_sd_x = egitim_orjinal['translation_x'].std()
        egitim_mean_y = egitim_orjinal['translation_y'].mean()
        egitim_sd_y = egitim_orjinal['translation_y'].std()

        # Tahmin yapılacak veriyi oluşturun
        input_data = pd.DataFrame({
            'onceki_x': [x],  # Bu değerleri kendi verinize göre değiştirin
            'onceki_y': [y]
        })

        # Veriyi R ortamına aktar
        r_input_data = pandas2ri.py2rpy(input_data)
        ro.globalenv['r_input_data'] = r_input_data

        # R modelini kullanarak tahmin yap
        predicted_x = ro.r('compute(YpySiniriAgi_model_x, r_input_data)$net.result')
        predicted_y = ro.r('compute(YpySiniriAgi_model_y, r_input_data)$net.result')

        # Tahmin sonuçlarını NumPy array'e dönüştür
        predicted_x_np = np.array(predicted_x).flatten()
        predicted_y_np = np.array(predicted_y).flatten()

        # Ters Z-score dönüşüm fonksiyonu
        def inverse_zscore(x, original_mean, original_sd):
            return (x * original_sd) + original_mean

        # Ters z-score uygulayarak tahminleri orijinal ölçeğe dönüştür
        predictions_original_x = inverse_zscore(predicted_x_np, egitim_mean_x, egitim_sd_x)
        predictions_original_y = inverse_zscore(predicted_y_np, egitim_mean_y, egitim_sd_y)

        # Sonuçları yazdır
        print(f"Orijinal ölçek tahmini translation_x: {predictions_original_x[0]}")
        print(f"Orijinal ölçek tahmini translation_y: {predictions_original_y[0]}")
        return predictions_original_x[0],predictions_original_y[0]