import numpy as np
import torch
import joblib

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.00001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = abs(val_loss)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score:
            self.counter += 1           
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True        
        elif score > self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.early_stop = False
        pass
        
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        joblib.dump(filename = path, value = model)
        self.val_loss_min = val_loss