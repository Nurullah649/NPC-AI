import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('GTK3Cairo')

coordinates = []
# Read and parse data from Sonuc2.txt
with open('/home/nurullah/Masaüstü/NPC-AI/data/Sonuc2.txt', 'r') as file:
    data = file.readlines()
    for line in data:
        line = line.strip().strip('[]')
        try:
            parts = line.split(',')
            x = float(parts[0])
            y = float(parts[1].split(']')[0])
            print(f"Translation X: {x}, Translation Y: {y}")
            coordinates.append((x/40.200010002181806, y/40.200010002181806))
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(e)

x_vals = [coord[0] for coord in coordinates]
y_vals = [coord[1] for coord in coordinates]

def calculate_E(x_hat, y_hat, x, y):
    N = len(x_hat)
    total_sum = 0

    for i in range(N):
        term = ((x_hat[i] - x[i]) ** 2 + (y_hat[i] - y[i]) ** 2) ** 0.5
        total_sum += term

    E = total_sum / N
    return E

# Read data from GT_Translations.csv file
gt_translations = pd.read_csv('/home/nurullah/Masaüstü/NPC-AI/data/GT_Translations.csv')

# Ensure both datasets have the same length
min_length = min(len(gt_translations), len(x_vals))
gt_translations = gt_translations[:min_length]
x_vals = x_vals[:min_length]
y_vals = y_vals[:min_length]

# Separate the x and y values from gt_translations
x_gt = gt_translations['translation_x'].values
y_gt = gt_translations['translation_y'].values

E = calculate_E(x_gt, y_gt, x_vals, y_vals)
print(E)
