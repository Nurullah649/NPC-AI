import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/nurullah/Masaüstü/tekno_server/tekno_server/GT_Translations.csv'  # replace with the actual file path
data = pd.read_csv(file_path)

# Extracting the columns
x = data['translation_x']
y = data['translation_y']

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Translation X')
plt.ylabel('Translation Y')
plt.title('Translation X vs Y')
plt.grid(True)
plt.savefig('translation_plot.png')

