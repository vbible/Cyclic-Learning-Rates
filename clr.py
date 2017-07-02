#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:35:19 2017

@author: victor
"""

import numpy as np

class CLR():
    def __init__(self, train_set, base_lr, max_lr, epochs_per_cycle = 2):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.epochs_per_cycle = epochs_per_cycle
        self.iterations_per_epoch = self.iterations_per_epoch(train_set)
        self.step_size = (self.epochs_per_cycle*self.iterations_per_epoch)/2

    def iterations_per_epoch(self, train_set):
        return np.ceil(train_set.dataset.__len__() / train_set.batch_size)
    
    def iteration(self, epoch, batch_idx):
        return epoch*self.iterations_per_epoch + batch_idx

    def lr(self, epoch, batch_idx):
        cycle = np.floor(1+self.iteration(epoch, batch_idx)/(2*self.step_size))
        x = np.abs(self.iteration(epoch, batch_idx)/self.step_size - 2*cycle + 1)
        lr = self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))
        return lr
