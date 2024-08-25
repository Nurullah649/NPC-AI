# We will define a class R_Model which will be used to train and predict using R models
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


class R_Model:
    def __init__(self,gt_data_path,alg_data_path):
        self.gt_data=pd.read_csv(gt_data_path)
        self.alg_data=pd.read_csv(alg_data_path)
        self.model=self.train_model(self.alg_data,self.gt_data)

    def train_model(self,gt_data,alg_data):
        return None
    def predict(self,alg_x,alg_y):
        pass