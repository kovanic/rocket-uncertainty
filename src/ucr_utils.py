import pandas as pd
import numpy as np
import torch

import os
from tqdm.auto import tqdm

from sktime.datasets import load_from_tsfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import OrdinalEncoder



def load_ucr_seq(path_to_ucr, seq, device):

    x_train, y_train = load_from_tsfile(path_to_ucr + f'{seq}/{seq}_TRAIN.ts', return_data_type="pd-multiindex")
    x_test, y_test = load_from_tsfile(path_to_ucr + f'{seq}/{seq}_TEST.ts', return_data_type="pd-multiindex")
    x_train = x_train.reset_index().pivot(index='instances', columns='timepoints', values='dim_0')
    x_test = x_test.reset_index().pivot(index='instances', columns='timepoints', values='dim_0')

    if x_train.shape[1] > x_train.shape[0]:
        x_test.loc[:, np.arange(x_test.shape[1], x_train.shape[1])] = np.nan
    elif x_train.shape[1] < x_train.shape[0]:
        x_train.loc[:, np.arange(x_train.shape[1], x_test.shape[1])] = np.nan


    x_train = x_train.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    x_train.interpolate(method='linear', axis=1, inplace=True)
    y_train = y_train[~x_train.isna().any(1)] # for deleting constant time series
    x_train.dropna(axis=0, inplace=True) # for deleting constant time series
    x_train = torch.from_numpy(x_train.values).unsqueeze(1).to(device)

    enc = OrdinalEncoder()
    y_train = enc.fit_transform(y_train[:, None]).ravel()
    y_train = torch.from_numpy(y_train.astype('int'))
    y_train = (y_train - y_train.min()).to(device)


    x_test = x_test.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    x_test.interpolate(method='linear', axis=1, inplace=True)
    y_test = y_test[~x_test.isna().any(1)]
    x_test.dropna(axis=0, inplace=True)
    x_test = torch.from_numpy(x_test.values).unsqueeze(1).to(device)

    y_test = enc.transform(y_test[:, None]).ravel()
    y_test = torch.from_numpy(y_test.astype('int'))
    y_test = (y_test - y_test.min()).to(device)

    return x_train, y_train, x_test, y_test


def train_val_split(x, y, val_size=0.2, seed=9):
    vals, counts = torch.unique(y, return_counts=True)
    val_size = np.maximum(len(vals)/len(y), val_size)
    #For classes, which have only one observation I use a noisy observation in train and original in validation
    if (counts == 1).any():
        one_obs_condition = (y == vals[counts==1].unsqueeze(1)).any(0)

        noisy_x = x[one_obs_condition] +  torch.empty_like(x[one_obs_condition]).normal_(0., 0.1)
        noisy_x = (noisy_x - noisy_x.mean()) / noisy_x.std()

        y_ = y[~one_obs_condition].cpu().numpy()
        if len(y_) > 0:
            idx_train, idx_val, *_ = train_test_split(np.arange(len(y_)), y_, test_size=val_size, random_state=seed, shuffle=True, stratify=y_)
        else:
            return noisy_x, y, x, y 
            
        x_train = x[~one_obs_condition][idx_train]
        x_train = torch.cat([x_train, noisy_x], dim=0)
        y_train = y[~one_obs_condition][idx_train]
        y_train = torch.cat([y_train, y[one_obs_condition]])

        x_val = x[~one_obs_condition][idx_val]
        x_val = torch.cat([x_val, x[one_obs_condition]], dim=0)
        y_val = y[~one_obs_condition][idx_val]
        y_val = torch.cat([y_val, y[one_obs_condition]])

    else:
        y_ = y.cpu().numpy()
        idx_train, idx_val, *_ = train_test_split(np.arange(len(y_)), y_, test_size=val_size, random_state=seed, shuffle=True, stratify=y_)
        x_train = x[idx_train]
        y_train = y[idx_train]
        x_val = x[idx_val]
        y_val = y[idx_val]

    return x_train, y_train, x_val, y_val
