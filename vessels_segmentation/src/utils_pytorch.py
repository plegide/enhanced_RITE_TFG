#Author: Álvaro S. Hervella

import torch
from torch.optim import Optimizer
import torch
from torch.utils.data.sampler import Sampler
from functools import partial
import csv
from typing import List, Optional, Tuple

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    
def load_model(model, filepath, strict=True):
    model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())), strict=strict)
    
def save_opt(opt, filepath):
    torch.save(opt.state_dict(), filepath)

def load_opt(opt, filepath):
    opt.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))
 

def save_to_csv(data, filepath):
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(data)


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, always in the same order.

    Arguments:
        indices (list): a list of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class IsBest:
    def __init__(self, lower_is_better=True):
        if lower_is_better:
            self.best_value = float('inf')
            self.check = self.check_lower_better
        else:
            self.best_value = float('-inf')
            self.check = self.check_higher_better

    def __call__(self, new_value):
        return self.check(new_value)

    def check_lower_better(self, new_value):
        best = False
        if new_value<=self.best_value:
            self.best_value = new_value
            best = True
        return best

    def check_higher_better(self, new_value):
        best = False
        if new_value>=self.best_value:
            self.best_value = new_value
            best = True
        return best
        

class EarlyStopping:
    def __init__(self, patience, min_delta, max_augm, last_patience):
        self.training = True
        self.wait = 0
        self.best_loss = 1e6
        self.min_delta = min_delta
        self.patience = patience
        self.last_patience = last_patience
        self.max_augm = max_augm
        
    def update(self, current_loss):
        dif = current_loss - self.best_loss
        if dif <= -self.min_delta:
            self.best_loss = current_loss
            self.training = True
            self.wait = 0
            return True
        else:
            if (self.wait >= self.patience and dif > self.max_augm) or (self.wait >= self.last_patience):
                self.training = False
            self.wait += 1
            return False
        
    def __call__(self):
        return self.training



class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, filepath, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
#        self.total_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
        self.last_lr = False
        self.train = True

        self.filepath = filepath
        
        save_to_csv([['epoch','new_lr']], self.filepath + '/scheduler.csv')
    
    def training(self):
        return self.train
        
    def set_patience(self, new_patience):
        self.patience = new_patience

    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
#        self.total_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
#        is_better = False

        if self.is_better(current, self.best):  #better or equal -> See function definition
            self.best = current
            self.num_bad_epochs = 0
#            self.total_bad_epochs = 0
            is_better = True
        else:
            self.num_bad_epochs += 1
#            self.total_bad_epochs += 1
            is_better = False

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        reduce_lr = False
        if self.num_bad_epochs > self.patience:
            reduce_lr = True
            self._reduce_lr(epoch)
            if self.last_lr:
                self.train = False
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
            
        return is_better, reduce_lr

    def _reduce_lr(self, epoch):
        last_lr = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    save_to_csv([[str(epoch),str(new_lr)]], \
                                 self.filepath + '/scheduler.csv')
                last_lr.append(False)
            else:
                last_lr.append(True)
        if all(last_lr):
            self.last_lr = True
        

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a <= best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a <= best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a >= best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a >= best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
