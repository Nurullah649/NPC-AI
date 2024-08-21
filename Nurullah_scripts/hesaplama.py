import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('GTK3Cairo')

coordinates = []
# Read and parse data from Combined_Sonuc.txt
with open('/home/nurullah/Desktop/NPC-AI/data/Sonuc_deneme_GPR.txt', 'r') as file:
    data = file.readlines()
    for line in data:
        try:
            parts = line.split(',')
            x = float(parts[0])
            y = float(parts[1])
            print(f"Translation X: {x}, Translation Y: {y}")
            coordinates.append((x, y))
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(e)

x_vals = [coord[0] for coord in coordinates]
y_vals = [coord[1] for coord in coordinates]


def calculate_E(x_hat, y_hat, x, y):
    if len(x_hat) != len(x) or len(y_hat) != len(y):
        raise ValueError("The lengths of the predicted and ground truth values do not match.")

    N = len(x_hat)
    total_sum = 0

    for i in range(N):
        term = ((x_hat[i] - x[i]) ** 2 + (y_hat[i] - y[i]) ** 2) ** 0.5
        total_sum += term

        # Print the term and its components for debugging
        if np.isnan(term) or np.isinf(term):
            print(f"Term calculation issue at index {i}:")
            print(f"x_hat[i]: {x_hat[i]}, y_hat[i]: {y_hat[i]}")
            print(f"x[i]: {x[i]}, y[i]: {y[i]}")
            print(f"Term: {term}")

    E = total_sum / N
    return E


# Read data from GT_Translations.csv file
gt_translations = pd.read_csv('/home/nurullah/Desktop/NPC-AI/data/2024_TUYZ_Online_Yarisma_Ana_Oturum.csv')

# Ensure the columns are numeric and handle NaN values
gt_translations = gt_translations[['translation_x', 'translation_y']].apply(pd.to_numeric, errors='coerce')
gt_translations = gt_translations.dropna()  # Drop rows with NaN values

# Separate the x and y values from gt_translations
x_gt = gt_translations['translation_x'].values
y_gt = gt_translations['translation_y'].values

# Ensure both datasets have the same length
min_length = min(len(gt_translations), len(x_vals))
gt_translations = gt_translations[:min_length]
x_vals = x_vals[:min_length]
y_vals = y_vals[:min_length]

# Remove NaN values from x_vals and y_vals
valid_indices = ~np.isnan(y_vals)  # Find indices where y_vals are not NaN
x_vals = np.array(x_vals)[valid_indices]
y_vals = np.array(y_vals)[valid_indices]
x_gt = x_gt[:len(x_vals)]  # Ensure x_gt length matches x_vals length
y_gt = y_gt[:len(y_vals)]  # Ensure y_gt length matches y_vals length

# Print lengths of arrays to ensure they match
print(f"Length of x_gt: {len(x_gt)}")
print(f"Length of y_gt: {len(y_gt)}")
print(f"Length of x_vals: {len(x_vals)}")
print(f"Length of y_vals: {len(y_vals)}")

# Check for NaN or Inf values in the arrays
print("Checking for NaN or Inf values in the arrays...")
for array_name, array in [("x_gt", x_gt), ("y_gt", y_gt), ("x_vals", x_vals), ("y_vals", y_vals)]:
    if np.any(np.isnan(array)):
        print(f"{array_name} contains NaN values.")
    if np.any(np.isinf(array)):
        print(f"{array_name} contains Inf values.")

# Calculate the error
try:
    E = calculate_E(x_gt, y_gt, x_vals, y_vals)
    print(f"Error E: {E}")
except ValueError as e:
    print(f"Error calculating E: {e}")
