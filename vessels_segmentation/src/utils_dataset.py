#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashvaro
"""
import random
import csv
import numpy as np
from skimage import transform, color
import torch
         
def read_samples_idx(filepath):
    with open(filepath, 'r') as file:
        reader = list(csv.reader(file))
        samples = [int(idx) for idx in reader[0]]
    return samples

def save_samples_experiments(samples_set, filepath):
    with open(filepath, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(samples_set)
    return

def read_samples_experiments(filepath):
    with open(filepath, 'r') as file:
        reader = list(csv.reader(file))
        experiments = [[int(x) for x in samples] for samples in reader]
    return experiments


def random_affine(npimages):
#    if random.random() < 0.5:
    h = npimages[0].shape[0]
    w = npimages[0].shape[1]

    # random rotations 
    dorotate = np.deg2rad(random.uniform(-90, 90))

    # random zooms
    zoom = 2**random.uniform(-0.35, 0.35) 

    # shearing
    shear = np.deg2rad(random.uniform(-20, 20))

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    center_shift   = np.array((h, w)) / 2.0
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(rotation = dorotate,
                                          scale =(1/zoom, 1/zoom),
                                          shear = shear)

    tform = tform_center + tform_aug + tform_uncenter
    
    return [transform.warp(img, tform, output_shape=img.shape, preserve_range=True, mode='constant', cval=0) for img in npimages]

    
def random_hsv_0(npimages):
    img = color.rgb2hsv(npimages[0])
    dh = random.uniform(-0.02,0.02)
    ds = random.uniform(-0.2,0.2)
    dv = random.uniform(-0.2,0.2)
    ms = random.uniform(0.8,1.2)
    mv = random.uniform(0.8,1.2)

    img[:,:,0] += dh
    img[:,:,0][img[:,:,0]>1] -= 1
    img[:,:,0][img[:,:,0]<0] += 1

    img[:,:,1] = np.clip(ms * (img[:,:,1] + ds), 0, 1)
    img[:,:,2] = np.clip(mv * (img[:,:,2] + dv), 0, 1)
    
    npimages[0] = color.hsv2rgb(img)
    
    return npimages

def random_hsv(npimages):
    img = color.rgb2hsv(npimages[0])
    dh = random.uniform(-0.02,0.02)
    es = 2**random.uniform(-1,1)
    ev = 2**random.uniform(-1,1)

    img[:,:,0] += dh
    img[:,:,0][img[:,:,0]>1] -= 1
    img[:,:,0][img[:,:,0]<0] += 1

    if random.random() < 0.5:
        img[:,:,1] = img[:,:,1]**es
    else:
        img[:,:,1] = 1-(1-img[:,:,1])**es

    if random.random() < 0.5:
        img[:,:,2] = img[:,:,2]**ev
    else:
        img[:,:,2] = 1-(1-img[:,:,2])**ev
    
    npimages[0] = color.hsv2rgb(img)
    
    return npimages  


def to_torch_tensors(npimages):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    return [torch.from_numpy(img.transpose(2,0,1).astype('float32')) for img in npimages]
        
def random_vertical_flip(npimages):
    if random.random() < 0.5:
        return [np.flip(img, axis=0) for img in npimages]
    else:
        return npimages
    
def random_horizontal_flip(npimages):
    if random.random() < 0.5:
        return [np.flip(img, axis=1) for img in npimages]
    else:
        return npimages

    
    
