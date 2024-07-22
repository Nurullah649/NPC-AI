import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import joblib  # veya import pickle

# CSV dosyasını okuma (drone verisi)
drone_data = pd.read_csv('data/GT_Translations.csv')

# TXT dosyasını okuma (algoritma çıktısı)
alg_data = []
with open('data/Sonuc2.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip().strip('[]')
        try:
            parts = line.split(',')
            tx = float(parts[0])
            ty = float(parts[1].split(']')[0])
            alg_data.append((tx, ty))
            print(f"Translation X: {tx}, Translation Y: {ty}")
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(e)

# DataFrame oluşturma
alg_data = pd.DataFrame(alg_data, columns=['alg_x', 'alg_y'])

# Verilerin birleştirilmesi
merged_data = pd.merge(drone_data, alg_data, left_index=True, right_index=True)
print(merged_data)

# Model için özellikler ve hedef değişkenler
X = merged_data[['translation_x', 'translation_y']]
y = merged_data[['alg_x', 'alg_y']]

# Polinom regresyon modeli oluşturma (derece 2)
degree = 2
model_x = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_y = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Modelleri eğitme
model_x.fit(X, y['alg_x'])
model_y.fit(X, y['alg_y'])

# Tahmin yapma
y_pred_x = model_x.predict(X)
y_pred_y = model_y.predict(X)

# Performans değerlendirmesi
mse_x = mean_squared_error(y['alg_x'], y_pred_x)
mse_y = mean_squared_error(y['alg_y'], y_pred_y)

print(f"Mean Squared Error for alg_x: {mse_x}")
print(f"Mean Squared Error for alg_y: {mse_y}")

# Modelleri kaydetme
joblib.dump(model_x, 'model_x.pkl')
joblib.dump(model_y, 'model_y.pkl')

# Daha sonra modelleri yüklemek için
# model_x = joblib.load('model_x.pkl')
# model_y = joblib.load('model_y.pkl')
