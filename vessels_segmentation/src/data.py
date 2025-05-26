#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashvaro
"""
import os
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import torch
import random

from collections import defaultdict
import re
from skimage import img_as_float


class VesselsDataset(Dataset):

    def __init__(self, input_dir, vessels_pattern, retino_pattern, mask_pattern, transform=None):
        self.input_path = input_dir
        self.vessels_pattern = vessels_pattern
        self.retino_pattern = retino_pattern
        self.mask_pattern = mask_pattern
        self.transform = transform
        self._make_dataset()

    def _make_dataset(self):        
        vessels = re.compile(self.vessels_pattern)
        retino = re.compile(self.retino_pattern)
        mask = re.compile(self.mask_pattern)   
        number = re.compile('[0-9]+')
        
        self.vessels = defaultdict(dict)
        self.retinos = defaultdict(dict)
        self.masks = defaultdict(dict)     
        
        for fname in os.listdir(self.input_path):
            n = number.findall(fname)            
            if n:
                n = int(n[0])
                if vessels.match(fname):
                    self.vessels[n]= fname
#                    print(fname)
                elif retino.match(fname):
                    self.retinos[n] = fname
#                    print(fname)
                elif mask.match(fname):
                    self.masks[n]= fname  
#                    print(fname)

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        retino = self.retinos[index]
        vessel = self.vessels[index]
        mask = self.masks[index]

        r = self._read_image_rgb(retino)
        m = self._read_image_hwc(mask)
        v = self._read_image_hwc(vessel)
        
        r = img_as_float(r)
        m = img_as_float(m)
        v = img_as_float(v)

        item = [r, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [index,item]
                        
    def _read_image(self, fname):
        return io.imread(os.path.join(self.input_path, fname)) #, as_gray=True)
    
    def _read_image_hwc(self, fname):
        #height, width and channels
        img = io.imread(os.path.join(self.input_path, fname)) #, as_gray=True)  
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
        else:
            img = img.transpose(1,2,0)
        return img
    
    def _read_image_rgb(self, fname):
        #They already have shape==3
        return io.imread(os.path.join(self.input_path, fname))
                    
    def drive_padding(npimages):
        return [np.pad(img, ((4,4),(5,6),(0,0)), mode='constant', constant_values=0) for img in npimages]
    
    def stare_padding(npimages):
        return [np.pad(img, ((1,2),(2,2),(0,0)), mode='constant', constant_values=0) for img in npimages]
    
    def remove_drive_padding(npimage):
        return npimage[4:-4,5:-6]

    def remove_stare_padding(npimage):
        return npimage[1:-2,2:-2]





