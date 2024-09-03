import os
import numpy as np
import matplotlib.pyplot as plt
from rpg_trajectory_evaluation import *
from rpg_trajectory_evaluation.trajectory import Trajectory

#traj = Trajectory('result')

# compute the absolute error
#print(traj.compute_absolute_error(),traj.compute_relative_errors(),traj.cache_current_error())

import pickle

# .pickle dosyasını okuma
with open('result/saved_results/traj_est/cached/cached_rel_err.pickle', 'rb') as file:
    data = pickle.load(file)

# Veriyi görüntüleme
for d in data:
    print(d)

# compute the relative error at sub-trajectory lengths computed from the whole trajectory length.


# save the relative error to `cached_rel_err.pickle`


# write the error statistics to yaml files


# static method to remove the cached error from a result folder
Trajectory.remove_cached_error('result')