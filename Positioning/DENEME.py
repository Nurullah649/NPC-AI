import os
# Set R environment variables
os.environ['R_HOME'] = r'C:/Program Files/R/R-4.4.1'
os.environ['PATH'] = r'C:/Program Files/R/R-4.4.1/bin/x64;' + os.environ['PATH']
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from sklearn.metrics import mean_absolute_error



# Load R libraries
neuralnet = importr('neuralnet')
base = importr('base')

# Load Pandas DataFrames


test_orjianl_x = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/normalized_test_data_x.csv")
test_orjianl_y = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/normalized_test_data_y.csv")
egitim_orjinal = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/egitim_orjinal.csv")
test_orjinal = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/test_orjinal.csv")

# Convert Pandas DataFrames to R data frames
pandas2ri.activate()

r_test_orjianl = pandas2ri.py2rpy(test_orjinal)

# Load R models
ro.r('load("C:/Users/sms/Documents/new_positioning_model/neuralnet_model_x_new.RData")')
ro.r('load("C:/Users/sms/Documents/new_positioning_model/neuralnet_model_y_new.RData")')

# Make predictions with the models
ro.globalenv['r_test_orjianl'] = test_orjinal
ro.globalenv['r_test_orjianl'] = test_orjinal
predictions_xtespit = ro.r('predict(YpySiniriAgi_model_x, r_test_orjianl)')
predictions_ytespit = ro.r('predict(YpySiniriAgi_model_y, r_test_orjianl)')

# Convert R predictions to NumPy arrays
predictions_xtespit_np = np.array(predictions_xtespit).flatten()
predictions_ytespit_np = np.array(predictions_ytespit).flatten()

# Load actual values from test set

test_df = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/test_orjinal.csv")

"""egitim_translation_x = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/egitim_translation_x.csv")
egitim_translation_y = pd.read_csv("C:/Users/sms/Documents/new_positioning_model/egitim_translation_y.csv")
"""
# Min and max values for inverse scaling
egitim_mean_x = egitim_orjinal['translation_x'].mean()
egitim_sd_x = egitim_orjinal['translation_x'].std()
egitim_mean_y = egitim_orjinal['translation_y'].mean()
egitim_sd_y = egitim_orjinal['translation_y'].std()
# Function for inverse scaling
def inverse_zscore(x, original_mean, original_sd):
    return (x * original_sd) + original_mean

# Inverse scale the predictions
predictions_original_x = inverse_zscore(predictions_xtespit_np, egitim_mean_x, egitim_sd_x)
predictions_original_y = inverse_zscore(predictions_ytespit_np, egitim_mean_y, egitim_sd_y)

# Combine predictions with actual values
results = test_df.copy()
results['predicted_x'] = predictions_original_x
results['predicted_y'] = predictions_original_y

# Set pandas display option to show more decimal places
pd.set_option('display.float_format', '{:.6f}'.format)

# Print results
print(results.head())

# Calculate Mean Absolute Error (MAE)
mae_x = mean_absolute_error(results['translation_x'], results['predicted_x'])
mae_y = mean_absolute_error(results['translation_y'], results['predicted_y'])

print(f"Mean Absolute Error (X): {mae_x}")
print(f"Mean Absolute Error (Y): {mae_y}")

# Save results to a CSV file
results.to_csv("C:/Users/sms/Documents/new_positioning_model/results.csv", index=False)

"""# Create prediction data frame in Python
pd1_veri = pd.DataFrame({

    'onceki_x': [0.377],
    'onceki_y': [0.0084]

})

# Convert the prediction data frame to an R data frame
r_pd1_veri = pandas2ri.py2rpy(pd1_veri)

# Assign the converted data frame to the R environment
ro.globalenv['r_pd1_veri'] = r_pd1_veri

# Perform the prediction using the R model
predicted_x = ro.r('predict(YpySiniriAgi_model_x, r_pd1_veri)')
predicted_y = ro.r('predict(YpySiniriAgi_model_y, r_pd1_veri)')

# Convert the prediction results back to NumPy arrays
predicted_x_np = np.array(predicted_x).flatten()
predicted_y_np = np.array(predicted_y).flatten()

# Print the predictions
print(f"Predicted translation_x: {predicted_x_np[0]}")
print(f"Predicted translation_y: {predicted_y_np[0]}")
"""