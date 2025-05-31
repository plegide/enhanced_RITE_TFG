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

    def __init__(self, input_dir, retino_pattern, geom_pattern, mask_pattern, transform=None):
        self.input_path = input_dir
        self.retino_pattern = retino_pattern
        self.geom_pattern = geom_pattern
        self.mask_pattern = mask_pattern
        self.transform = transform
        self._make_dataset()

    def _make_dataset(self):        
        retinos = re.compile(self.retino_pattern)
        geom = re.compile(self.geom_pattern)
        mask = re.compile(self.mask_pattern)   
        number = re.compile('[0-9]+')
        
        self.retinos = defaultdict(dict)
        self.geom_maps = defaultdict(dict)
        self.masks = defaultdict(dict)     
        
        for fname in os.listdir(self.input_path):
            n = number.findall(fname)            
            if n:
                n = int(n[0])
                if retinos.match(fname):
                    self.retinos[n] = fname
                elif geom.match(fname):
                    self.geom_maps[n] = fname
                elif mask.match(fname):
                    self.masks[n] = fname

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        retino = self.retinos[index]
        geom_map = self.geom_maps[index]
        mask = self.masks[index]

        r = self._read_image_rgb(retino)  # Entrada RGB
        m = self._read_image_hwc(mask)
        
        try:
            geom_data = np.load(os.path.join(self.input_path, geom_map))
            maxima = geom_data['maxima'].astype(np.float32)[:,:,np.newaxis]
            displacement = geom_data['displacement'].astype(np.float32)
            radius = geom_data['radius'].astype(np.float32)[:,:,np.newaxis]
        except Exception as e:
            print(f"Error loading geometric map {geom_map}: {str(e)}")
            raise
    
        r = img_as_float(r)
        m = img_as_float(m)

        item = [r, m, maxima, displacement, radius]  # r es la entrada RGB
        if self.transform is not None:
            item = self.transform(item)
        return [index, item]
                        
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





