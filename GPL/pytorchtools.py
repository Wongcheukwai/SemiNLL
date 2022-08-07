import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, model_path='', optimizer_path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_pre_max = np.Inf
        self.delta = delta
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.trace_func = trace_func

    def __call__(self, model_pre, model, optimizer):

        score = model_pre

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model_pre, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model_pre, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, model_pre, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'model_pre increased ({self.model_pre_max:.3f} --> {model_pre:.3f}).  Saving model ...\n')
        torch.save(model.state_dict(), self.model_path)
        torch.save(optimizer.state_dict(), self.optimizer_path)
        self.model_pre_max = model_pre
