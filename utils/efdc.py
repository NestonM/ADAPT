import os
import subprocess
import numpy as np
import pandas as pd
from shutil import copyfile
import joblib
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import _utils
buoys = [2477, 3569, 2150, 2783, 6827] # row number of cells for assimilation
from exp.exp_Aquaformer import *
from utils.load import *
fixes = ['1.020', '1.022', '1.024', '1.026', '1.028', '1.030', '1.035', '1.040', '1.045', '1.050'] # candidates for KTR
pres = ['1-', '1.25-', '1.5-'] # candidates for KRO
KRO_values = [1, 1.25, 1.5]
KTR_values = [1.02, 1.022, 1.024, 1.026, 1.028, 1.03, 1.035, 1.04, 1.045, 1.05]
sites = ['314', 'CJH', 'HYH', 'QYY'] # names of assimilation sites
Lys = [1, 6, 48, 144] # forecasting horizons

def run_efdc(model_path: str, efdc_exe: str):
    print('efdc running')
    subprocess.run(["cmd.exe", "/c", 'run.bat'], check=True)
    pass

def wq_reader(buoys, path_file):
    with open(path_file, 'r') as f:
        lines = f.readlines()
        lines_1 = [float(lines[buoys[0]].split()[-3]), float(lines[buoys[1]].split()[-3]), float(lines[buoys[2]].split()[-3]), float(lines[buoys[3]].split()[-3]), float(lines[buoys[4]].split()[-3])]
        lines_2 = [float(lines[buoys[0]+1].split()[-3]), float(lines[buoys[1]+1].split()[-3]), float(lines[buoys[2]+1].split()[-3]), float(lines[buoys[3]+1].split()[-3]), float(lines[buoys[4]+1].split()[-3])]
        lines_3 = [float(lines[buoys[0]+2].split()[-3]), float(lines[buoys[1]+2].split()[-3]), float(lines[buoys[2]+2].split()[-3]), float(lines[buoys[3]+2].split()[-3]), float(lines[buoys[4]+2].split()[-3])]
    output = []
    for i in range(len(buoys)):
        output.append((lines_1[i]+lines_2[i]+lines_3[i])/3)
    return output

def wqs_reader(model_path='./EE/Model_run', buoys=[2477, 3569, 2150, 2783, 6827]):
    path_file = model_path+'/#output/WQWCRST.OUT'
    wq = np.array(wq_reader(buoys, path_file))
    return wq

def COD_reader(buoys, path_file):
    with open(path_file, 'r') as f:
        lines = f.readlines()
        lines_1 = [float(lines[buoys[0]].split()[-4]), float(lines[buoys[1]].split()[-4]), float(lines[buoys[2]].split()[-4]), float(lines[buoys[3]].split()[-4]), float(lines[buoys[4]].split()[-4])]
        lines_2 = [float(lines[buoys[0]+1].split()[-4]), float(lines[buoys[1]+1].split()[-4]), float(lines[buoys[2]+1].split()[-4]), float(lines[buoys[3]+1].split()[-4]), float(lines[buoys[4]+1].split()[-4])]
        lines_3 = [float(lines[buoys[0]+2].split()[-4]), float(lines[buoys[1]+2].split()[-4]), float(lines[buoys[2]+2].split()[-4]), float(lines[buoys[3]+2].split()[-4]), float(lines[buoys[4]+2].split()[-4])]
    output = []
    for i in range(len(buoys)):
        output.append((lines_1[i]+lines_2[i]+lines_3[i])/3)
    return output

def CODs_reader(model_path='./EE/Model_run', buoys=[2477, 3569, 2150, 2783, 6827]):
    path_file = model_path+'/#output/WQWCRST.OUT'
    wq = np.array(COD_reader(buoys, path_file))
    return wq

def stamp_reader(model_path='./EE/Model_run', buoys=[2477, 3569, 2150, 2783, 6827]):
    pass

def one_forecast(exp, pre, n_fix, n_site, Ly, inp, pred_flag=1):
    # inp is an array of shape (Lx, 2)
    feature = np.expand_dims(inp, axis=0)
    label = np.ones((1, Ly, inp.shape[-1]))
    sites_PSR = joblib.load('./PSR/sites_PSR.save')
    lags, d = sites_PSR[pre][n_fix][n_site], sites_PSR['m']
    ds, dl = tensor_input(Ly, feature, label, 1, lags, d, pred_flag, input_flag=[0, 1], max_d=40)
    lstm = joblib.load("./results/{}/v_{}/{}/model_{}_l".format(sites[n_site], pre+fixes[n_fix], '6', 'Aquaformer')) # 6 could be replaced by any horizon with trained LSTM ready in the path
    exp.PSR = [lags[-1], d]
    if Ly==1:
        preds = []
        for feature, label in dl:
            pred, _ = lstm(feature, feature, feature)
            preds.append(pred.squeeze(-1))
            torch.cuda.empty_cache()
        preds_np = torch.cat(preds, dim=0).detach().cpu().numpy()            
        N, L = preds_np.shape            
        pred_tile = np.tile(preds_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)            
        scaler = joblib.load('./Scaler/scaler_{}_{}.save'.format(pre+fixes[n_fix], sites[n_site]))            
        pred_inv = scaler.inverse_transform(pred_tile)
        pred = pred_inv[:, pred_flag].reshape(N, L)
    else:
        exp.model = joblib.load("./results/{}/v_{}/{}/model_{}".format(sites[n_site], pre+fixes[n_fix], Ly, 'Aquaformer'))
        exp.args.site_target = sites[n_site]
        _, pred = exp.predict(pre, fixes[n_fix], lstm, dl, update=0)
    return pred

def grid_forecast(exp, Ly, inps, pred_flag=1):
    sites_results = {}
    for pre in pres:
        sites_results[pre] = [0]*len(fixes)
        for i in range(len(fixes)):
            sites_results[pre][i] = []
            for j in range(len(sites)):
                sites_results[pre][i].append(one_forecast(exp, pre, i, j, Ly, inps[j]))
            sites_results[pre][i] = np.concatenate(sites_results[pre][i], axis=0)
    return sites_results

def horizon_forecast(exp, inps, Ly_flag=[0,1,2,3]):
    results = {}
    for i in Ly_flag:
        Ly = Lys[i]
        results[str(Ly)] = grid_forecast(exp, Ly, inps)
    return results

def df_loader(results, labels, Ly_flag = [0, 1, 2, 3]):
    if Ly_flag == [0]:
        Lys = [1, 1, 1, 1]
    else:
        Lys = [1, 6, 48, 144]
    data = []
    for k in range(len(pres)):
        for i in range(len(fixes)):
            data.append({
                'KRO': KRO_values[k],
                'KTR': KTR_values[i],
                'pred_1': results[str(Lys[0])][pres[k]][i],
                'pred_6': results[str(Lys[1])][pres[k]][i],
                'pred_48': results[str(Lys[2])][pres[k]][i],
                'pred_144': results[str(Lys[3])][pres[k]][i],
                'label_1': labels['1'],
                'label_6': labels['6'],
                'label_48': labels['48'],
                'label_144': labels['144'],
                'error_1': labels['1'],
                'error_6': labels['6'],
                'error_48': labels['48'],
                'error_144': labels['144']
            })
    df = pd.DataFrame(data)
    return df

# Function to calculate distance penalty
def calculate_distance_penalty(row, previous, lambda_p):
    kro_diff = 0.01 * (row['KRO'] - previous['KRO'])
    ktr_diff = row['KTR'] - previous['KTR']
    distance = np.sqrt(kro_diff**2 + ktr_diff**2)
    penalty = lambda_p * distance  # Linear penalty
    return penalty

# Apply softmax to scores to get joint probabilities
def softmax(x, kappa=1):
    x = np.array(x)
    k = (x-np.min(x))/(np.max(x)-np.min(x))*kappa
    e_x = np.exp(k)  # for numerical stability
    return e_x / e_x.sum()

def error_cal(df, weights, DA_flag=[0, 1, 2, 3], Ly_flag=[0, 1, 2, 3]):
    errors = []
    df_c = df.copy()
    df_c['Final_Error'] = df_c['KRO'] - df_c['KRO']
    for i in Ly_flag:
        Ly = Lys[i]
        p, l, e = 'pred_'+str(Ly), 'label_'+str(Ly), 'error_'+str(Ly)
        errors = []
        for row_index in range(len(df.index)):
            w = weights[str(Ly)][row_index][DA_flag] # for a combination, site*Ly has a map of weights
            error = ((abs(df_c[p][row_index][DA_flag]-df_c[l][row_index][DA_flag])/df_c[l][row_index][DA_flag]).mean(axis=-1)*w).sum()
            errors.append(error)
        df_c[e] = errors
        df_c['Final_Error'] = df_c['Final_Error'] + df_c[e]
    df = df_c.copy()
    return df

def prob_cal(df, previous_optimal = {'KRO': 1.0, 'KTR': 1.024}):
    # Define lambda for distance penalty
    lambda_penalty = 0.2  # Adjust based on experimentation

    df['Penalty'] = df.apply(
        calculate_distance_penalty,
        axis=1,
        previous=previous_optimal,
        lambda_p=lambda_penalty
    )

    # Compute adjusted scores: negative final error minus penalty
    df['Score'] = -df['Final_Error'] - df['Penalty']
    df['Probability'] = softmax(df['Score'].values)
    return df

def K_switch_group(df):
    # Compute marginal distributions
    marginal_kro = df.groupby('KRO')['Probability'].sum().reset_index()
    marginal_ktr = df.groupby('KTR')['Probability'].sum().reset_index()

    mu_kro = (marginal_kro['KRO'] * marginal_kro['Probability']).sum()
    mu_ktr = (marginal_ktr['KTR'] * marginal_ktr['Probability']).sum()

    # Round expected values to nearest grid points
    optimal_kro = min(KRO_values, key=lambda x: abs(x - mu_kro))
    optimal_ktr = min(KTR_values, key=lambda x: abs(x - mu_ktr))
    previous_optimal = {'KRO': optimal_kro, 'KTR': optimal_ktr}
    return previous_optimal

def K_switch(df):
    index = df['Probability'].idxmax()
    optimal_kro = df['KRO'][index]
    optimal_ktr = df['KTR'][index]
    optimal = {'KRO': optimal_kro, 'KTR': optimal_ktr}
    return optimal

def efdc_time(origin_forward, period, model_path='./EE/Model_run'):
    if origin_forward != 0:
        with open(model_path+'/efdc.inp', 'r') as f:
            lines = f.readlines()
            C8 = [float(n) for n in lines[215].split()]
            C8[1] += 0.0208333*origin_forward
            lines[215]= " ".join(str(i) for i in C8)+'\n'
        with open(model_path+'/efdc.inp', 'w') as f:
            for line in lines:
                f.writelines(line)
    if period != 0:
        with open(model_path+'/efdc.inp', 'r') as f:
            lines = f.readlines()
            C7 = [float(n) for n in lines[197].split()]
            C7[0] = period
            lines[197]= " ".join(str(i) for i in C7)+'\n'
        with open(model_path+'/efdc.inp', 'w') as f:
            for line in lines:
                f.writelines(line)
    if origin_forward==1:
        copyfile(model_path+'/#output/WQWCRST.OUT', model_path+'/wq.inp')
        copyfile(model_path+'/#output/RESTART.OUT', model_path+'/re.inp')
        copyfile(model_path+'/#output/RSTWD.OUT', model_path+'/rs.inp')
        print('one step copied')
        copyfile(model_path+'/#output/WQWCRST.OUT', model_path+'/wqwcrst.inp')
        copyfile(model_path+'/#output/RESTART.OUT', model_path+'/restart.inp')
        copyfile(model_path+'/#output/RSTWD.OUT', model_path+'/rstwd.inp')
        print('forward copied')
    else:
        copyfile(model_path+'/#output/WQWCRST.OUT', model_path+'/wqwcrst.inp')
        copyfile(model_path+'/#output/RESTART.OUT', model_path+'/restart.inp')
        copyfile(model_path+'/#output/RSTWD.OUT', model_path+'/rstwd.inp')
        print('forward copied')
    pass

def efdc_setup(optimal, model_path='./EE/Model_run'):
    copyfile(model_path+'/wq.inp', model_path+'/wqwcrst.inp')
    copyfile(model_path+'/re.inp', model_path+'/restart.inp')
    copyfile(model_path+'/rs.inp', model_path+'/rstwd.inp')
    with open(model_path+'/wq3dwc.inp', 'r') as f:
        lines = f.readlines()
        K = [float(n) for n in lines[516].split()]
        K[2], K[3] = optimal['KRO'], optimal['KTR']
        lines[516]= " ".join(str(i) for i in K)+'\n'
    with open(model_path+'/wq3dwc.inp', 'w') as f:
        for line in lines:
            f.writelines(line)
    pass

def efdc_dump(model_path, stamp):
    copyfile(model_path+'/wq.inp', model_path+'/wq_{}.inp'.format(stamp))
    copyfile(model_path+'/re.inp', model_path+'/re_{}.inp'.format(stamp))
    copyfile(model_path+'/rs.inp', model_path+'/rs_{}.inp'.format(stamp))
    copyfile(model_path+'/efdc.inp', model_path+'/efdc_{}.inp'.format(stamp))
    copyfile(model_path+'/wq3dwc.inp', model_path+'/wq3dwc_{}.inp'.format(stamp))
    pass

def efdc_resume(model_path, stamp):
    copyfile(model_path+'/wq_{}.inp'.format(stamp), model_path+'/wqwcrst.inp')
    copyfile(model_path+'/re_{}.inp'.format(stamp), model_path+'/restart.inp')
    copyfile(model_path+'/rs_{}.inp'.format(stamp), model_path+'/rstwd.inp')
    copyfile(model_path+'/efdc_{}.inp'.format(stamp), model_path+'/efdc.inp')
    copyfile(model_path+'/wq3dwc_{}.inp'.format(stamp), model_path+'/wq3dwc.inp')
    pass