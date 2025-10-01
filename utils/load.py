import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import _utils
from utils.PSR import *
import joblib
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pres = ['1-', '1.25-', '1.5-']
fixes = ['1.020', '1.022', '1.024', '1.026', '1.028', '1.030', '1.035', '1.040', '1.045', '1.050']

def dataset_loader(load_flag):
    if load_flag == 'train':
        sites_train_classic = joblib.load('./data/sites_train_classic')
        sites_train_label = joblib.load('./data/sites_train_label')
        sites_valid_classic = joblib.load('./data/sites_valid_classic')
        sites_valid_label = joblib.load('./data/sites_valid_label')
        return sites_train_classic, sites_train_label, sites_valid_classic, sites_valid_label
    elif load_flag == 'PSR':
        sites_r = joblib.load('./data/sites_PSR')
        return sites_r

def reconstruction(data, m, tau):
    n = data.shape[0]
    m = m.shape[0]
    tau = tau.shape[0]
    M = n - (m-1) * tau
    Data = torch.FloatTensor(m, M).to(device)
    for j in range(M):
        for i in range(m):
            Data[i, j] = data[i*tau+j]
    return Data # (m, M)

def PSR_params_input(data_PSR, params):
    max_lag, bins, max_dim = params[0], params[1], params[2]
    N = int(len(data_PSR))
    _, d = data_PSR[0].shape
    lags = []
    ds = []
    start = time.time()
    for i in range(d):
        taus = np.zeros(N)
        ms = np.zeros(N)
        lags_i, ds_i = MICM([data_PSR[j][:, i] for j in range(N)], max_lag, bins, max_dim)
        lags.append(np.array(lags_i))
        ds.append(np.array(ds_i))
        print(np.array(lags_i), np.array(ds_i))
    lags = np.concatenate(lags)
    ds = np.concatenate(ds)
    np.save('./PSR/lags', lags)
    np.save('./PSR/ds', ds)
    print(f"time taken:{time.time()-start}")
    # lags, ds (4*N)
    return lags, ds

def tensor_input(pred_len, feature, label, batch_size, lags, d, pred_flag, input_flag=[0, 1], max_d=40):
    feature, label = feature.transpose(2, 0, 1), label.transpose(2, 0, 1)
    feature_np, label_np = feature[input_flag], label[input_flag] # (4, N, L) 
    _, _, win_f = label_np.shape
    prime = win_f - pred_len
    feature, label = torch.tensor(feature_np, dtype = torch.float).to(device), torch.tensor(label_np, dtype = torch.float).to(device)
    del feature_np, label_np
    size = feature.shape[1] - feature.shape[1] % batch_size
    x, y = feature[:, :size, :].permute(1, 2, 0), label[:, :size, :].permute(1, 2, 0)
    data = torch.concat([x, y], dim=1)
    N, L, _ = data.shape
    lag_max = int(np.max(lags))
    x_re = []
    y_re = []
    for j in range(len(lags)):
        lag = int(lags[j])
        M = L - (d - 1) * lag
        trim = (lag_max - lag) * (d - 1)
        PSR = torch.concat([data[:, i:i+(d-1)*lag+1:lag, j].reshape(N, 1, -1) for i in range(int(M))], dim = 1)
        x_PSR = PSR[:, trim:-win_f, :] # (N, M, d_WQ*m) now d_WQ is 1 when specific
        if prime == 0:
            y_PSR = PSR[:, (-win_f):, :]
        elif pred_len == 1:
            y_PSR = PSR[:, -win_f, :].unsqueeze(1)
        else:
            y_PSR = PSR[:, (-win_f):(-prime), :] # (N, M, d_WQ*m)
        x_re.append(x_PSR)
        y_re.append(y_PSR)
        del x_PSR, y_PSR, PSR
    x_re, y_re = torch.concat(x_re, dim=-1), torch.concat(y_re, dim=-1)
    ds = TensorDataset(x_re, y_re) # (N, M, d_WQ*m)
    del feature, label, size, x, y, x_re, y_re
    dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, num_workers = 0)
    return ds, dl
    
def transfer_input(pred_len, pre, n_fix, n_site, sites_trim, pars, batch_size_src, site_target, pred_flag, input_flag, params, load_flag, tc=0, tl=0, vc=0, vl=0):
    if load_flag == 'PSR':
        sites_r = dataset_loader(load_flag)
        data_PSR = []
        for pre in pres:
            for i in range(len(fixes)):
                for j in range(len(sites_trim)):
                    data_PSR.append(sites_r[pre][i][j])
        if os.path.isfile('./PSR/lags.npy') and os.path.isfile('./PSR/ds.npy'):
            lags, ds = np.load('./PSR/lags.npy'), np.load('./PSR/ds.npy')
        else:
            lags, ds = PSR_params_input(data_PSR, params)
        sites_PSR = {}
        for k in range(len(pres)):
            sites_PSR[pres[k]] = [0]*len(fixes)
            for i in range(len(fixes)):
                sites_PSR[pres[k]][i] = [0]*len(sites_trim)
                for j in range(len(sites_trim)):
                    sites_PSR[pres[k]][i][j] = np.array([lags[k*len(fixes)*len(sites_trim)+i*len(sites_trim)+j], lags[k*len(fixes)*len(sites_trim)+i*len(sites_trim)+j+len(pres)*len(fixes)*len(sites_trim)]])
        sites_PSR['m'] = stats.mode(ds)[0][0]
        joblib.dump(sites_PSR, './PSR/sites_PSR.save')
        print(sites_PSR)
        return sites_PSR
    
    if load_flag == 'train':
        sites_train_classic, sites_train_label, sites_valid_classic, sites_valid_label = tc, tl, vc, vl
        if os.path.isfile('./PSR/sites_PSR.save'):
            sites_PSR = joblib.load('./PSR/sites_PSR.save')
            print(['sites_PSR loaded'])
        else:
            print('PSR file not found')
            sites_PSR = {}

        tr_ds = []
        tr_dl = []
        val_ds = []
        val_dl = []
        lags, d = sites_PSR[pre][n_fix][n_site], sites_PSR['m']
        tr_ds, tr_dl = tensor_input(pred_len, sites_train_classic[str(pred_len)][pre][n_fix][n_site], sites_train_label[str(pred_len)][pre][n_fix][n_site], batch_size_src, lags, d, pred_flag, input_flag)
        val_ds, val_dl = tensor_input(pred_len, sites_valid_classic[str(pred_len)][pre][n_fix][n_site], sites_valid_label[str(pred_len)][pre][n_fix][n_site], batch_size_src, lags, d, pred_flag, input_flag)
        return tr_dl, val_dl, sites_PSR