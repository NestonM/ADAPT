import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import _utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

from utils.load import *
from utils.tools import *
from models.Aquaformer import *

class Exp_Aquaformer(object):
    def __init__(self, args, PSR):
        self.args = args
        self.PSR = PSR
    
    def _build_model(self):
        model_dict = {
            'Aquaformer':Aquaformer,
        }
        model_l = Aqua_lstm(self.PSR, self.args).float()
        model_a = model_dict[self.args.model](
            self.PSR,
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.device
        ).float()
        model = [model_l, model_a]
        optimizer = [torch.optim.Adam(model_l.parameters(), lr = self.args.learning_rate), torch.optim.Adam(model_a.parameters(), lr = self.args.learning_rate)]
        scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimizer[0], gamma = self.args.gamma), torch.optim.lr_scheduler.ExponentialLR(optimizer[1], gamma = self.args.gamma)]

        self.model = model
        return self.model, optimizer, scheduler
    
    def _train_step(self, pre, fix, lstm, dl, optimizer, scheduler, transfer_flag, update):
        model = self.model
        model.train()
        losses = []
        r2s = []
        encs = []
        decs = []
        preds = []
        labels = []
        pred_flag = self.args.pred_flag
        site_target = self.args.site_target
        print('iteration = {}'.format(len(dl)))
        
        if not model.training:
            model.train()
            print('eval -> train')
        else:
            print('train')
        
        for feature, label in dl:
            N, M, _ = label.shape
            _, _, dM = feature.shape
            zeros = torch.zeros((N, M, dM)).to(device)
            x_dec = torch.cat([feature, zeros], dim=1)
            PSR, d = self.PSR, self.args.enc_in
            lag, m = PSR[0], PSR[1]
            model.train()
            pred, attns = model(feature, x_dec, lstm, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
            preds.append(pred.squeeze(-1))
            label = label[:, :, (pred_flag+1)*m-1]
            loss = self.args.loss_function(pred.squeeze(-1), label)
            labels.append(label)
            del feature, label, pred, x_dec
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            del loss

        preds_np = torch.cat(preds, dim=0).detach().cpu().numpy()
        labels_np = torch.cat(labels, dim=0).detach().cpu().numpy()
            
        N, L = preds_np.shape
            
        pred_tile = np.tile(preds_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)
        label_tile = np.tile(labels_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)
            
        scaler = joblib.load('./Scaler/scaler_{}_{}.save'.format(pre+fix, self.args.site_target))
            
        pred_inv = scaler.inverse_transform(pred_tile)
        label_inv = scaler.inverse_transform(label_tile)
            
        MAPE = np.mean(abs(pred_inv[:, pred_flag].flatten() - label_inv[:, pred_flag].flatten()) / label_inv[:, pred_flag].flatten())
        pred_inv, label_inv = pred_inv[:, pred_flag].reshape(N, L), label_inv[:, pred_flag].reshape(N, L)
        loss_mean = np.mean(losses)

        MSE = mean_squared_error(label_inv, pred_inv)
        MAE = mean_absolute_error(label_inv, pred_inv)
        print('MSE=', MSE, 'MAE=', MAE)
        loss_mean = np.mean(losses)
        del losses, preds, labels
        self.model = model
        return self.model, pred_inv, label_inv, loss_mean, MAPE
    
    def _train_model(self, pre, fix, dl, tgt_dl, tests_dl, transfer_flag, update, lstm_flag):
        run_name = self.args.run_name
        pred_flag = self.args.pred_flag
        pred_flag_test = self.args.pred_flag
        site_target = self.args.site_target
        model, optimizer, scheduler = self._build_model()
        lstm = 0
        if lstm_flag=='lstm':
            optimizer = optimizer[0]
            scheduler = scheduler[0]
            lstm = []
            self.model = model[0]
        else:
            optimizer = optimizer[1]
            scheduler = scheduler[1]
            lstm = joblib.load("./results/{}/v_{}/{}/model_{}_l".format(site_target, pre+fix, self.args.pred_len, self.args.model))
            self.model = model[1]
        site_target = self.args.site_target       
        
        batch_size_tgt = self.args.batch_size_tgt

        best = self.model
        loss = 1
        time_start = time.time()
        preds = []
        labels = []
        losses = []
        loss_check = self.args.loss_check
        run_name = self.args.run_name
        pred_flag = self.args.pred_flag
        pred_flag_test = self.args.pred_flag
        site_target = self.args.site_target
        early_stopping = EarlyStopping(patience=self.args.patience, delta = self.args.delta, verbose=True)
        for epoch in range(self.args.epoch_num):
            self.model.train()
            with torch.enable_grad():
                self.model, pred, label, loss, MAPE = self._train_step(pre, fix, lstm, dl, optimizer, scheduler, transfer_flag, update)           
            print('epoch = {} | loss = {} | MAPE = {}'.format(epoch + 1, loss, MAPE))
            losses.append(loss)
            preds.append(pred)
            labels.append(label)
            if lstm_flag=='lstm':
                path = "./results/{}/v_{}/{}/model_{}_l".format(site_target, pre+fix, self.args.pred_len, self.args.model)
                loss_es = loss
            else:
                path = "./results/{}/v_{}/{}/model_{}".format(site_target, pre+fix, self.args.pred_len, self.args.model)
                label_inv, pred_inv = self.predict(pre, fix, lstm, tgt_dl, update)
                self.model.train()
                loss_es = math.sqrt(mean_squared_error(label_inv, pred_inv))
            early_stopping(loss_es, self.model, path)                      
            
            if epoch % 10 == 9:
                time_end = time.time()
                time_sum = time_end - time_start
                print('epoch = {} | loss = {} | MAPE = {} | time = {}'.format(epoch + 1, loss, MAPE, time_sum))
                time_start = time.time()
            
            if abs(loss) < abs(loss_check):
                best = self.model
                print('optimal loss = {}'.format(loss))
                break
            if self.args.pred_flag != 3:
                if MAPE < 0.01:
                    best = self.model
                    print('optimal loss = {}'.format(loss))
                    break
            else:
                if MAPE < 0.005:
                    best = self.model
                    print('optimal loss = {}'.format(loss))
                    break
            del loss, pred, label
            if early_stopping.early_stop and epoch > 20:
                print("Early stopping")
                break
            scheduler.step()
        del best
        return losses, preds, labels
    
    def train(self, pre, fix, dl, tgt_dl, test_dl, transfer_flag, update, lstm_flag):
        site_target = self.args.site_target
        pred_flag_test = self.args.pred_flag
        if not os.path.exists("./results/{}/v_{}".format(site_target, pre+fix)):
            os.mkdir("./results/{}/v_{}".format(site_target, pre+fix))
        if not os.path.exists("./results/{}/v_{}/{}".format(site_target, pre+fix, self.args.pred_len)):
            os.mkdir("./results/{}/v_{}/{}".format(site_target, pre+fix, self.args.pred_len))
        losses, preds, labels = self._train_model(pre, fix, dl, tgt_dl, test_dl, transfer_flag, update, lstm_flag)
        return losses, preds, labels
        
    def predict(self, pre, fix, lstm, dl, update, k=80):
        site_target = self.args.site_target
        window_future = self.args.pred_len
        pred_flag = self.args.pred_flag
        win_h = self.args.seq_len
        model = self.model
        model.eval()
        if self.args.model=='Aquaformer':
            lstm.eval()
        preds = []
        labels = []
        PSR, d = self.PSR, self.args.enc_in
        lag, m = PSR[0], PSR[1]
        
        with torch.no_grad():
            for feature, label in dl:
                N, M, _ = label.shape
                _, _, dM = feature.shape
                zeros = torch.zeros((N, M, dM)).to(device)
                x_dec = torch.cat([feature, zeros], dim=1)
                PSR, d = self.PSR, self.args.enc_in
                lag, m = PSR[0], PSR[1]
                pred, attns = model(feature, x_dec, lstm, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                preds.append(pred.squeeze(-1))
                labels.append(label[:, :, (pred_flag+1)*m-1].squeeze(-1))
                del feature, label, pred, x_dec, zeros, attns
                torch.cuda.empty_cache()
    
        self.model.train()
        lstm.train()
        preds_np = torch.cat(preds, dim=0).detach().cpu().numpy()
        labels_np = torch.cat(labels, dim=0).detach().cpu().numpy()
            
        N, L = preds_np.shape
            
        pred_tile = np.tile(preds_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)
        label_tile = np.tile(labels_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)
            
        scaler = joblib.load('./Scaler/scaler_{}_{}.save'.format(pre+fix, self.args.site_target))
            
        pred_inv = scaler.inverse_transform(pred_tile)
        label_inv = scaler.inverse_transform(label_tile)
            
        MAPE = np.mean(abs(pred_inv[:, pred_flag].flatten() - label_inv[:, pred_flag].flatten()) / label_inv[:, pred_flag].flatten())
        pred_inv, label_inv = pred_inv[:, pred_flag].reshape(N, L), label_inv[:, pred_flag].reshape(N, L)
        
        MSE = mean_squared_error(label_inv, pred_inv)
        MAE = mean_absolute_error(label_inv, pred_inv)
        r2 = r2_score(label_inv, pred_inv)
        print('MSE=', MSE, 'MAE=', MAE, 'MAPE=', MAPE, 'r2=', r2)
        del preds, labels        
        return label_inv, pred_inv
        
    def predict_transfer(self, lstm, dl, update, k=80):
        site_target = self.args.site_target
        window_future = self.args.pred_len
        pred_flag = self.args.pred_flag
        win_h = self.args.seq_len
        model = self.model
        model.eval()
        lstm.eval()
        preds = []
        labels = []
        PSR, d = self.PSR, self.args.enc_in
        lag, m = PSR[0], PSR[1]
        
        with torch.no_grad():
            for feature, label in dl:
                N, M, _ = label.shape
                _, _, dM = feature.shape
                #zeros = feature[:, -1, :].unsqueeze(1).repeat(1, M, 1)
                zeros = torch.zeros((N, M, dM)).to(device)
                x_dec = torch.cat([feature, zeros], dim=1)
                PSR, d = self.PSR, self.args.enc_in
                lag, m = PSR[0], PSR[1]
                pred, attns = model(feature, x_dec, lstm, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                preds.append(pred.squeeze(-1))
                labels.append(label[:, :, (pred_flag+1)*m-1].squeeze(-1))
                del feature, label, pred, x_dec, zeros, attns
                torch.cuda.empty_cache()
    
        self.model.train()
        lstm.train()
        preds_np = torch.cat(preds, dim=0).detach().cpu().numpy()
        labels_np = torch.cat(labels, dim=0).detach().cpu().numpy()
            
        N, L = preds_np.shape
            
        pred_tile = np.tile(preds_np.reshape(N, L, 1), (1, 1, 4)).reshape(-1, 4)
        label_tile = np.tile(labels_np.reshape(N, L, 1), (1, 1, 4)).reshape(-1, 4)
            
        scaler = joblib.load('./Scaler/scaler_{}_{}.save'.format(site_target, update))
            
        pred_inv = scaler.inverse_transform(pred_tile)
        label_inv = scaler.inverse_transform(label_tile)
            
        MAPE = np.mean(abs(pred_inv[:, pred_flag].flatten() - label_inv[:, pred_flag].flatten()) / label_inv[:, pred_flag].flatten())
        pred_inv, label_inv = pred_inv[:, pred_flag].reshape(N, L), label_inv[:, pred_flag].reshape(N, L)
        
        MSE = mean_squared_error(label_inv, pred_inv)
        MAE = mean_absolute_error(label_inv, pred_inv)
        r2 = r2_score(label_inv, pred_inv)
        print('MSE=', MSE, 'MAE=', MAE, 'MAPE=', MAPE, 'r2=', r2)
        del preds, labels        
        return label_inv, pred_inv
        
    def predict_error(self, pre, fix, lstm, dl, update, k=80):
        site_target = self.args.site_target
        window_future = self.args.pred_len
        pred_flag = self.args.pred_flag
        win_h = self.args.seq_len
        model = self.model
        model.eval()
        lstm.eval()
        preds = []
        labels = []
        PSR, d = self.PSR, self.args.enc_in
        lag, m = PSR[0], PSR[1]
        
        with torch.no_grad():
            for feature, label in dl:
                N, M, _ = label.shape
                _, _, dM = feature.shape
                zeros = torch.zeros((N, M, dM)).to(device)
                x_dec = torch.cat([feature, zeros], dim=1)
                PSR, d = self.PSR, self.args.enc_in
                lag, m = PSR[0], PSR[1]
                pred, attns = model(feature, x_dec, lstm, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
                preds.append(pred.squeeze(-1))
                labels.append(label[:, :, (pred_flag+1)*m-1].squeeze(-1))
                del feature, label, pred, x_dec, zeros, attns
                torch.cuda.empty_cache()
    
        self.model.train()
        lstm.train()
        preds_np = torch.cat(preds, dim=0).detach().cpu().numpy()
        labels_np = torch.cat(labels, dim=0).detach().cpu().numpy()
            
        N, L = preds_np.shape
            
        pred_tile = np.tile(preds_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)
        label_tile = np.tile(labels_np.reshape(N, L, 1), (1, 1, 2)).reshape(-1, 2)
            
        scaler = joblib.load('./Scaler/scaler_{}_{}.save'.format(pre+fix, self.args.site_target))
            
        pred_inv = scaler.inverse_transform(pred_tile)
        label_inv = scaler.inverse_transform(label_tile)
            
        MAPE = np.mean(abs(pred_inv[:, pred_flag].flatten() - label_inv[:, pred_flag].flatten()) / label_inv[:, pred_flag].flatten())
        pred_inv, label_inv = pred_inv[:, pred_flag].reshape(N, L), label_inv[:, pred_flag].reshape(N, L)
        
        MSE = mean_squared_error(label_inv, pred_inv)
        MAE = mean_absolute_error(label_inv, pred_inv)
        r2 = r2_score(label_inv, pred_inv)
        print('MSE=', MSE, 'MAE=', MAE, 'MAPE=', MAPE, 'r2=', r2)
        del preds, labels
        
        return MSE, MAE, MAPE, r2