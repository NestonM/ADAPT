import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import _utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time
import argparse

parser = argparse.ArgumentParser(description='[ADAPT] Analytical Data Assimilation based on Phase-space Tuning')

parser.add_argument('--model', type=str, default='Aquaformer',help='model of experiment')   
parser.add_argument('--site_target', type=str, default='XMT_', help='target site')
parser.add_argument('--sites_trim', type=str, default='314,CJH,HYH,QYY,XMT', help='site list')
parser.add_argument('--pars', type=str, default='T,DO', help='param list')
parser.add_argument('--input_flag', type=str, default='0,1', help='input param ID')
parser.add_argument('--pred_flag', type=int, default='1', help='relative pred param ID')
parser.add_argument('--DA_flag', type=str, default='0,1,2,3,4', help='sites used for DA')
parser.add_argument('--Ly_flag', type=str, default='0', help='horizons used for DA')
parser.add_argument('--seq_len', type=int, default=144, help='input sequence length of Informer encoder')
parser.add_argument('--pred_len', type=int, default=144, help='prediction sequence length')
parser.add_argument('--n_sets', type=int, default=4, help='number of updating')
parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=4, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of lstm')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--e_freeze', type=int, default=2, help='num of encoder layers frozen')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_freeze', type=int, default=2, help='num of decoder layers frozen')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--lstm_layers', type=int, default=2, help='num of lstm layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=True)
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--delta', type=float, default=0.00001, help='early stopping delta')
parser.add_argument('--epoch_num', type=int, default=60, help='train epochs')
parser.add_argument('--batch_size_src', type=int, default=125, help='batch size of source data')
parser.add_argument('--batch_size_tgt', type=int, default=12, help='batch size of target data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--gamma', type=float, default=0.98, help='scheduler gamma')
parser.add_argument('--learning_rate_tune', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--gamma_tune', type=float, default=0.97, help='scheduler gamma')
parser.add_argument('--loss_function', type=str, default='mse',help='loss function')
parser.add_argument('--loss_check', type=float, default=0.0001, help='loss checkpoint')
parser.add_argument('--r2_check', type=float, default=0.8, help='r2 checkpoint')
parser.add_argument('--run_name', type=str, default='1122_01',help='run name')
parser.add_argument('--limit', type=int, default=200, help='pred sequence num')
parser.add_argument('--k', type=int, default=80, help='pred curve num')
parser.add_argument('--tau', type=int, default=5, help='PSR tau delay')
parser.add_argument('--m', type=int, default=8, help='PSR added dimension')
parser.add_argument('--PSR_flag', type=int, default=1, help='PSR calculation flag')
parser.add_argument('--PSR_params', type=str, default='100,300,6', help='max_lag, bins, max_dim')
parser.add_argument('--optimal', type=str, default='1.25,1.024', help='KRO, KTR')
parser.add_argument('--src_index', type=int, default=1, help='source scaler index')


parser.add_argument('--device', type=str, default='cuda:0',help='device id of gpu or cpu')
args = parser.parse_args(args=['--site_target', '314', 
                               '--pred_flag', '1', 
                               '--model', 'Aquaformer',
                               '--pred_len', '48',
                               '--run_name', 'DAPS', 
                               '--input_flag', '0,1', 
                               '--e_layers', '5',
                               '--d_layers', '4', 
                               '--learning_rate', '0.0001',
                               '--gamma', '0.97',
                               '--learning_rate_tune', '0.001',
                               '--gamma_tune', '0.8',
                               '--d_model', '64',
                               '--batch_size_src', '125', 
                               '--batch_size_tgt', '6', 
                               '--patience', '5', 
                               '--epoch_num', '100', '--limit', '200'])

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.input_flag = [int(flag) for flag in args.input_flag.replace(' ','').split(',')]
args.sites_trim = [site for site in args.sites_trim.replace(' ','').split(',')]
args.DA_flag = [int(flag) for flag in args.DA_flag.replace(' ','').split(',')]
args.Ly_flag = [int(flag) for flag in args.Ly_flag.replace(' ','').split(',')]
args.optimal = [float(par) for par in args.optimal.replace(' ','').split(',')]
args.optimal = {'KRO':args.optimal[0], 'KTR':args.optimal[1]}
args.pars = [par for par in args.pars.replace(' ','').split(',')]
args.enc_in = len(args.input_flag)
args.dec_in = len(args.input_flag)
args.device = torch.device(args.device)
args.loss_function = nn.MSELoss() if args.loss_function=='mse' else nn.CrossEntropyLoss()
args.PSR_params = [int(par) for par in args.PSR_params.replace(' ','').split(',')]

print('Args in experiment:')
print(args)

from exp.exp_Aquaformer import *
from exp.exp_DA import *
from utils.load import *

PSR = [12, 4] # delay time, embedding dimension
exp = Exp_Aquaformer(args, PSR)
exp_DA = Exp_DA(args)
dfs, DOs, CODs, valids, optimals = exp_DA.DA(exp)
results_sites = [dfs, DOs, CODs, valids, optimals]
joblib.dump(results_sites, './results/results_sites_{}'.format(args.DA_flag))