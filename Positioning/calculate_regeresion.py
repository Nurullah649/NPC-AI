import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.linear_model import LinearRegression

# Load data from updated_video_data.txt
coordinates = []
with open('../../DPVO/iptal_sim.txt', 'r') as file:
    data = file.readlines()
    for line in data:
        try:
            print(data)
            parts = line.split(',')
            x = float(parts[0])  # x value
            y = float(parts[1])  # y value
            coordinates.append([x, y])
        except (ValueError, IndexError):
            continue  # Skip invalid lines


# Separate x and y values
x_vals = [coord[0] for coord in coordinates]
y_vals = [coord[1] for coord in coordinates]

# Load ground truth translations from CSV
gt_translations = pd.read_csv('../content/2024_TUYZ_İptal_Online_Yarisma.csv')
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
model = LinearRegression(fit_intercept=False, positive=False)
model.fit(y_train, x_train)
scale_factor = model.coef_
offset = model.intercept_


# Predict remaining translations
predictions = []
for x, y in zip(x_gt[:train_size], y_gt[:train_size]):  # Corrected loop
    predictions.append([x, y])
for x, y in zip(x_vals[train_size:], y_vals[train_size:]):
    scaled_positions = np.dot((x, y), scale_factor.T) + offset
    print(scaled_positions)
    pred_translation_x = scaled_positions[0]
    pred_translation_y = scaled_positions[1]
    predictions.append([pred_translation_x, pred_translation_y])


# Convert predictions to a NumPy array for slicing
predictions = np.array(predictions)

# Write results to a text file
with open('combined_results.txt', 'w') as output_file:
    for i in range(len(predictions)):
        output_file.write(f"{predictions[i][0]} {predictions[i][1]}\n")

# Plotting combined data
plt.figure(figsize=(10, 6))

# Plot the ground truth values (all of them)
plt.scatter(x_gt, y_gt, color='blue', label='Ground Truth', marker='o')

# Plot the combined data (first 450 ground truth + predictions)
plt.scatter(predictions[:, 0], predictions[:, 1], color='red', label='Ground Truth & Predictions', marker='x')

# Labeling the plot
plt.title('Ground Truth vs Combined Data (Ground Truth & Predictions)')
plt.xlabel('Translation X')
plt.ylabel('Translation Y')
plt.legend()

# Show the plot
plt.show()