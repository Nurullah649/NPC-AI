import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the GT_Translations.csv file
gt_translations = pd.read_csv('/home/nurullah/Desktop/NPC-AI/content/2024_Ornek_Veri_GT.csv')

# Initialize an empty list to store the parsed data from Sonuc2.txt
coordinates = []

# Read and parse the data from the Sonuc2.txt file
with open('/home/nurullah/Desktop/DPVO/result/images.txt', 'r') as file:
    data = file.readlines()
    for line in data:
        try:
            parts = line.split()
            if len(parts) >= 8:  # 8. eleman x, 9. eleman y
                x = float(parts[5])  # x değeri
                y = float(parts[6])  # y değeri
            print(f"Translation X: {x}, Translation Y: {y}")
            coordinates.append((-x, -y))
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(e)

# Convert the parsed data to a DataFrame
sonuc2_translations = pd.DataFrame(coordinates, columns=['translation_x', 'translation_y'])

# Ensure both datasets have the same length
min_length = min(len(gt_translations), len(sonuc2_translations))
gt_translations = gt_translations[:min_length]
sonuc2_translations = sonuc2_translations[:min_length]

# Calculate the scale factor
gt_distances = np.sqrt(gt_translations['translation_x']**2 + gt_translations['translation_y']**2)
sonuc2_distances = np.sqrt(sonuc2_translations['translation_x']**2 + sonuc2_translations['translation_y']**2)
scale_factor = np.mean(sonuc2_distances / gt_distances)

# Calculate the angular difference
gt_angles = np.arctan2(gt_translations['translation_y'], gt_translations['translation_x'])
sonuc2_angles = np.arctan2(sonuc2_translations['translation_y'], sonuc2_translations['translation_x'])
angle_difference = np.mean(sonuc2_angles - gt_angles)

# Convert the angular difference to degrees
angle_difference_degrees = np.degrees(angle_difference)

# Plot the data points
plt.figure(figsize=(10, 6))
plt.scatter(gt_translations['translation_x'], gt_translations['translation_y'], color='blue', label='GT Translations')
plt.scatter(sonuc2_translations['translation_x'], sonuc2_translations['translation_y'], color='red', label='Sonuc Translations')

# Adding labels and title
plt.xlabel('Translation X')
plt.ylabel('Translation Y')
plt.title(f'Translations Comparison\nScale Factor: {scale_factor:.2f}, Angular Difference: {angle_difference_degrees:.2f} degrees')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

print(f"Scale Factor: {scale_factor}")
print(f"Angular Difference: {angle_difference_degrees} degrees")
