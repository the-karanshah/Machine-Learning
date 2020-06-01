# import libraries
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import scipy.io

data_path = "./"
def get_data(folder, name):
    y = pd.read_csv(folder + name + "_Labels.csv", index_col = 'Id')
    x = pd.DataFrame(pd.read_pickle(folder + name + "_Features.pkl")).T
    print('loaded', x.shape, y.shape)
    return x,y
def pre_process_data(x,y):
#   x['one'] = int(1)
  y = y.reindex(x.index)
  x  = np.array(x)
#   x = x / np.linalg.norm(x, axis=0)
  y = np.array(y)
  return x,y


train_x, train_y = get_data(data_path, 'Train')
val_x, val_y = get_data(data_path, 'Val') 

scipy.io.savemat(data_path + 'q2_kaggle_data.mat', mdict={
    'trD': train_x,
    'trLb': train_y,
    'valD': val_x,
    'valLb': val_y
})