import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from images.txt
coordinates = []
with open('/home/nurullah/DPVO/result/images.txt', 'r') as file:
    data = file.readlines()
    print(len(data))
    for line in data:
        try:
            parts = line.split()
            x = float(parts[5])  # x value
            y = float(parts[6])  # y value
            coordinates.append((x, y))
        except (ValueError, IndexError):
            print('Hata')
            continue  # Skip invalid lines
print(len(coordinates))
# Separate x and y values
x_vals = [coord[0] for coord in coordinates]
y_vals = [coord[1] for coord in coordinates]

# Load ground truth translations from CSV
gt_translations = pd.read_csv('../content/2024_TUYZ_Online_Yarisma.csv')
gt_translations = gt_translations[['translation_x', 'translation_y']].apply(pd.to_numeric, errors='coerce').dropna()

# Extract x and y values
x_gt = gt_translations['translation_x'].values
y_gt = gt_translations['translation_y'].values

# Ensure both datasets are the same length
min_length = min(len(gt_translations), len(x_vals))
print(min_length)
x_vals = np.array(x_vals[:min_length])
y_vals = np.array(y_vals[:min_length])
x_gt = x_gt[:min_length]
y_gt = y_gt[:min_length]

# Use the first 450 points for linear regression
train_size = 450
x_train = np.column_stack((x_gt[:train_size], y_gt[:train_size]))
y_train = np.column_stack((x_vals[:train_size], y_vals[:train_size]))

# Train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict remaining translations
x_test = np.column_stack((x_gt[train_size:], y_gt[train_size:]))
predictions = model.predict(x_test)

# Combine original ground truth and predictions
combined_data = np.vstack((
    np.column_stack((x_gt, y_gt)),
    np.column_stack((x_test[:, 0], x_test[:, 1]))
))

# Write results to a text file
with open('combined_results.txt', 'w') as output_file:
    for i in range(len(combined_data)):
        output_file.write(f"{combined_data[i][0]} {combined_data[i][1]}\n")
