import os
import subprocess
import numpy as np
import pandas as pd
import joblib
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import _utils
from exp.exp_Aquaformer import *
from utils.load import *
from utils.efdc import *

class Exp_DA(object):
    def __init__(self, args):
        self.args = args
        self.optimal = args.optimal
        self.DA_flag = args.DA_flag
        self.Ly_flag = args.Ly_flag
        self.labels = joblib.load('./data/labels_run')
        self.weights = joblib.load('./data/weights_run')
        self.temps = joblib.load('./data/temps_run')
        self.inp = joblib.load('./data/inp_run')
        self.DOs = []
        self.CODs = []
        self.valids = {'DOs':[], 'CODs':[]}
        self.optimals = []
        self.dfs = []
        self.model_path = './EE/Model_run/'
        self.resume = 0
        self.resume_flag = False
        self.debug = False
    
    def _DA_step(self, stamp, exp, model_path='V:\DA\DAPS\EE\Model_run', efdc_exe='S:\Program Files (x86)\DSI\EEMS8.3\EFDCPlus_083_OMP_171229.exe'):
        if stamp % 24 == 0:
            df = prob_cal(error_cal(df_loader(horizon_forecast(exp, self.inp, self.Ly_flag), self.labels[stamp], self.Ly_flag), self.weights, self.DA_flag, self.Ly_flag), self.optimal)
            self.optimal = K_switch(df)
            self.optimals.append(self.optimal)    
        new_inp = np.concatenate([self.temps[stamp+1][:, -1].reshape(-1, 1), self.labels[stamp]['1']], axis=-1).reshape(-1, 1, 2)
        self.inp = np.concatenate([self.inp, new_inp], axis=1)[:, 1:, :]
        if stamp % 24 == 0:
            return df
        else:
            pass 
    
    def _DA_run(self, exp):
        steps = len(self.labels)
        n = int(steps / 24)
        print(n)
        for i in range(1, n+1):
            if self.resume_flag == True and os.path.isfile('./results/results_sites_{}_{}'.format(self.args.DA_flag, i*24)):
                resume = i*24
                results_sites = joblib.load('./results/results_sites_{}_{}'.format(self.args.DA_flag, i*24))
                self.dfs, self.optimals = results_sites[0], results_sites[1]
            else:
                resume = -1
        self.resume = resume
        for i in range(resume+1, steps):                      
            time_start = time.time()
            if i % 24 == 0:
                self.dfs.append(self._DA_step(i, exp))
                results_sites = [self.dfs, self.optimals]
                joblib.dump(results_sites, './results/results_sites_{}_{}'.format(self.args.DA_flag, i))
            else:
                self._DA_step(i, exp)
            time_end = time.time()
            time_sum = time_end - time_start
            print('epoch = {} time = {} optimal: {}'.format(i, time_sum, self.optimals[-1]))            
        pass
    
    def DA(self, exp):
        print(self.DA_flag, self.Ly_flag)
        self._DA_run(exp)
        return self.dfs, self.optimals

class Exp_EFDC(object):
    def __init__(self, args):
        self.args = args
        self.optimal = args.optimal
        self.DA_flag = args.DA_flag
        self.Ly_flag = args.Ly_flag
        self.labels = joblib.load('./data/labels_run')
        self.weights = joblib.load('./data/weights_run')
        self.temps = joblib.load('./data/temps_run')
        self.inp = joblib.load('./data/inp_run')
        self.DOs = []
        self.CODs = []
        self.valids = {'DOs':[], 'CODs':[]}
        self.optimals = joblib.load('./results/results_sites_[0, 1, 2, 3, 4]')[1]
        self.dfs = []
        self.model_path = './EE/Model_run/'
        self.resume = 0
        self.resume_flag = False
        self.debug = False
    
    def _DA_step(self, stamp, exp, model_path='V:\DA\DAPS\EE\Model_run', efdc_exe='S:\Program Files (x86)\DSI\EEMS8.3\EFDCPlus_083_OMP_171229.exe'):
        # the paths should be updated according to the actual situation
        if not self.debug == True:
            if self.resume_flag == True:
                efdc_resume(self.model_path, self.resume)
                self.resume_flag = False
            run_efdc(model_path, efdc_exe)
            DO = wqs_reader(self.model_path)
            self.valids['DOs'].append(wqs_reader(self.model_path, buoys_valid))
            self.valids['CODs'].append(CODs_reader(self.model_path, buoys_valid))
            self.DOs.append(wqs_reader(self.model_path))
            self.CODs.append(CODs_reader(self.model_path))
            efdc_time(1, 0, self.model_path)
            if stamp % 24 == 0:
                efdc_time(0, 5, self.model_path)
                run_efdc(model_path, efdc_exe)
                self.valids['DOs'].append(wqs_reader(self.model_path, buoys_valid))
                self.valids['CODs'].append(CODs_reader(self.model_path, buoys_valid))
                self.DOs.append(wqs_reader(self.model_path))
                self.CODs.append(CODs_reader(self.model_path))
                efdc_time(5, 42, self.model_path)
                run_efdc(model_path, efdc_exe)
                self.valids['DOs'].append(wqs_reader(self.model_path, buoys_valid))
                self.valids['CODs'].append(CODs_reader(self.model_path, buoys_valid))
                self.DOs.append(wqs_reader(self.model_path))
                self.CODs.append(CODs_reader(self.model_path))
                efdc_time(42, 96, self.model_path)
                run_efdc(model_path, efdc_exe)
                self.valids['DOs'].append(wqs_reader(self.model_path, buoys_valid))
                self.valids['CODs'].append(CODs_reader(self.model_path, buoys_valid))
                self.DOs.append(wqs_reader(self.model_path))
                self.CODs.append(CODs_reader(self.model_path))
                efdc_time(-47, 1, self.model_path)
                efdc_dump(self.model_path, stamp)
        else:
            DO = np.ones(5)
            
        new_inp = np.concatenate([self.temps[stamp+1][:, -1].reshape(-1, 1), DO.reshape(-1, 1)], axis=-1).reshape(-1, 1, 2)
        self.inp = np.concatenate([self.inp, new_inp], axis=1)[:, 1:, :]

        if stamp % 24 == 0:
            df = prob_cal(error_cal(df_loader(horizon_forecast(exp, self.inp), self.labels[stamp], self.Ly_flag), self.weights, self.DA_flag, self.Ly_flag), self.optimal)
            self.optimal = K_switch(df)
            self.optimals.append(self.optimal)
            efdc_setup(self.optimal, self.model_path)        
            return df
        else:
            pass
    
    def _DA_run(self, exp):
        steps = len(self.labels)
        n = int(steps / 24)
        print(n)
        for i in range(1, n+1):
            if self.resume_flag == True and os.path.isfile('./results/results_sites_{}_{}'.format(self.args.DA_flag, i*24)):
                resume = i*24
                results_sites = joblib.load('./results/results_sites_{}_{}'.format(self.args.DA_flag, i*24))
                self.dfs, self.DOs, self.CODs, self.valids, self.optimals = results_sites[0], results_sites[1], results_sites[2], results_sites[3], results_sites[4]
            else:
                resume = -1
        self.resume = resume
        for i in range(resume+1, steps):                      
            if i % 24 == 0:
                time_start = time.time()
                self.dfs.append(self._DA_step(i, exp))
                time_end = time.time()
                time_sum = time_end - time_start
                print('time = {}'.format(time_sum))
                results_sites = [self.dfs, self.DOs, self.CODs, self.valids, self.optimals]
                joblib.dump(results_sites, './results/results_sites_{}_{}'.format(self.args.DA_flag, i))
            else:
                time_start = time.time()
                self._DA_step(i, exp)
                time_end = time.time()
                time_sum = time_end - time_start
                print('time = {}'.format(time_sum))
        pass
    
    def DA(self, exp):
        print(self.DA_flag)
        tail = ''
        for flag in self.DA_flag:
            self.model_path = self.model_path + str(flag)
            tail += str(flag)
        with open('./run.bat', 'rt') as bat_file:
            text = bat_file.readlines()
            text[0]= f'CD "V:\DA\DAPS\EE\Model_run\{tail}\"'+'\n' # the absolute path of the project folder
        with open('./run.bat', 'wt') as bat_file:
            for line in text:
                bat_file.write(line)
        self._DA_run(exp)
        return self.dfs, self.DOs, self.CODs, self.valids, self.optimals