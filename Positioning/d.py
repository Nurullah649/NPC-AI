import pandas as pd

# Load the data
filename = 'Result_2.txt'
data = pd.read_csv(filename, header=None)

# Generate a timestamp based on the index or some time interval
time_interval = 1.0  # Assume 1 second between each entry
data.insert(0, 'timestamp', [i * time_interval for i in range(len(data))])

# Add the tz and quaternion values
data.columns = ['timestamp', 'tx', 'ty']
data['tz'] = 0.0
data['qx'] = 0.0
data['qy'] = 0.0
data['qz'] = 0.0
data['qw'] = 1.0

# Reorder and save
data = data[['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']]
output_filename = 'result/stamped_traj_estimate.txt'
data.to_csv(output_filename, header=False, index=False, sep=' ')
