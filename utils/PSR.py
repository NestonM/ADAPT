import numpy as np
import time
import pandas as pd
import sys
from scipy.stats import entropy
from scipy.fftpack import fft
from scipy import fftpack
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import _utils
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reconstruction_np(data, m, tau):
    n = data.shape[0]
    M = n - (m-1) * tau
    Data = np.zeros([m, M])
    for j in range(M):
        for i in range(m):
            Data[i, j] = data[i*tau+j]
    return Data # (m, M)

def mutual_information(data, max_lag, bins):
    mi = np.zeros(max_lag)
    N = len(data)
    lags = np.zeros(N)
    for n in range(N):
        series = data[n]
        for lag in range(1, max_lag + 1):
            # Create lagged version of series
            y1 = series[:-lag]
            y2 = series[lag:]

            # Compute joint histogram
            joint_hist, _, _ = np.histogram2d(y1, y2, bins=bins)
            # Normalize to get joint probability distribution
            p_y1y2 = joint_hist / np.sum(joint_hist)
            p_y1 = np.sum(p_y1y2, axis=1)[np.nonzero(np.sum(p_y1y2, axis=1))]
            p_y2 = np.sum(p_y1y2, axis=0)[np.nonzero(np.sum(p_y1y2, axis=0))]
            p_y1y2 = p_y1y2[np.nonzero(p_y1y2)]
            
            H_y1 = -np.sum(p_y1 * np.log(p_y1)/np.log(2))
            H_y2 = -np.sum(p_y2 * np.log(p_y2)/np.log(2))
            H_y1y2 = -np.sum(p_y1y2 * np.log(p_y1y2)/np.log(2))

            # Calculate mutual information
            mi[lag - 1] = H_y1 + H_y2 - H_y1y2
        plt.plot(mi)
        plt.show()
        low = argrelextrema(mi, np.less, order=4)[0][0] + 1
        print(n, low)
        if low > 40 or (low - lags[n-n%4] >= 5 and n%4>0 and low - lags[n-1] >= 5):
            low = argrelextrema(mi, np.less, order=3)[0][0] + 1
            print('order=3', low)
            if low > 40 or (low - lags[n-n%4] >= 5 and n%4>0 and low - lags[n-1] >= 5):
                low = argrelextrema(mi, np.less, order=2)[0][0] + 1
                print('order=2', low)
        elif low < 10 or (lags[n-n%4] - low >= 5 and n%4>0 and lags[n-1] - low >= 5):
            low = argrelextrema(mi, np.less, order=5)[0][0] + 1
            print('order=5', low)
            if low < 10 or (lags[n-n%4] - low >= 5 and n%4>0 and lags[n-1] - low >= 5):
                low = argrelextrema(mi, np.less, order=15)[0][0] + 1
                print('order=15', low)
                if low > 40 or (low - lags[n-n%4] >= 5 and n%4>0 and low - lags[n-1] >= 5):
                    low = argrelextrema(mi, np.less, order=10)[0][0] + 1
                    print('order=10', low)
                    if low > 40 or (low - lags[n-n%4] >= 5 and n%4>0 and low - lags[n-1] >= 5):
                        low = argrelextrema(mi, np.less, order=8)[0][0] + 1
                        print('order=8', low)            
        else:
            print('order=4', low)
        lags[n] = low       
    
    return lags

#Cao's method to determine embedding dimension
def nearest_neighbor_distances(X_m_raw, time_delay):
    n_points = X_m_raw.shape[0] # X_m (M, m)
    MM = int(n_points-time_delay)
    X_m = X_m_raw[:MM]
    distances = []
    dist_index = []
    for i in range(MM):
        # Exclude the point itself in distance calculation
        dist = X_m[np.arange(MM) != i] - X_m[i]
        norm = np.linalg.norm(dist, axis=1, ord=np.inf)
        if np.max(norm) == 0:
            print('norm = 0')
            break
        for j in range(len(norm)):
            if np.sort(norm)[j] > 0:
                #print(np.sort(norm)[j])
                distances.append(np.sort(norm)[j])
                dist_index.append(np.argsort(norm)[j])
                break 
    return np.array(distances), np.array(dist_index) # (MM)

def reconstruct(series, dim, time_delay):
        N = len(series)
        M = int(N - (dim - 1) * time_delay)
        if N - (dim - 1) * time_delay <= 0:
            return np.empty((0, dim))
        reconstructed = np.concatenate([series[i:int(i+(dim-1)*time_delay)+1:int(time_delay)].reshape(1, -1) for i in range(M)], axis=0)
        return reconstructed
    
def cao_method(series, max_dim, time_delay):
    E1s = []
    E2s = []
    count1 = count2 = 0
    emb_m1 = emb_m2 = emb_m = 0
    max_dim = min(max_dim, len(series)/time_delay-2)
    for m in range(2, int(max_dim)+1):
        if m % 10 == 0:
            print(m)
        X_m = reconstruct(series, m, time_delay)
        X_m1 = reconstruct(series, m + 1, time_delay)
        X_m2 = reconstruct(series, m + 2, time_delay)

        distances_m, dist_index_m = nearest_neighbor_distances(X_m, time_delay)
        distances_m1, dist_index_m1 = nearest_neighbor_distances(X_m1, time_delay)
        distances_m2, dist_index_m2 = nearest_neighbor_distances(X_m2, time_delay)
        
        E_m = np.mean(distances_m1 / distances_m[:len(distances_m1)])
        E_m1 = np.mean(distances_m2 / distances_m1[:len(distances_m2)])
        EE_m = np.mean([abs(X_m[1:, -1][i] - X_m[1:, -1][dist_index_m[i]]) for i in range(len(distances_m)-1)])
        EE_m1 = np.mean([abs(X_m1[1:, -1][i] - X_m1[1:, -1][dist_index_m1[i]]) for i in range(len(distances_m1)-1)])

        # Compute E1 and E2
        E1 = E_m1 / E_m
        E2 = EE_m1 / EE_m
        
        E1s.append(E1)
        E2s.append(E2)        
        
        # Check if E1 and E2 have stabilized
        if m > 2:
            if abs(E1s[m-2]-E1s[m-3]) < 0.08:  # Thresholds for E1 and E2
                count1 += 1
                if count1 == 1:
                    emb_m1 = m - 1
            elif abs(E1s[m-2]-E1s[m-3]) >= 0.08 and emb_m1 != 0:  # Thresholds for E1 and E2
                count1 = emb_m1 = 0
        if m > 1 and abs(E2s[m-2]-1) > 0.05:
            count2 += 1
        if count1 >= 2 and count2 > 0:
            emb_m = max(emb_m1, emb_m2)
            print(emb_m)
            break
    if emb_m == 0:
        print('random series', count1, count2)
    plt.plot(E1s)
    plt.plot(E2s)
    plt.show()

    return emb_m

def MICM(data, max_lag, bins, max_dim):
    lags = mutual_information(data, max_lag, bins)
    if not os.path.isfile('./PSR/lags.npy'):
        np.save('./PSR/lags', lags)
    dims = []
    for i in range(len(lags)):
        if i%4 == 0:
            print(i)
        emb_m = cao_method(data[i], max_dim, lags[i])

        if emb_m != 0:
            dims.append(int(emb_m))
    return lags, dims