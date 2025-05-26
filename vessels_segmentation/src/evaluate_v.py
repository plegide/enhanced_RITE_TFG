#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashvaro
"""
import torch
from PIL import Image
import torchvision.transforms as tr
from sklearn import metrics
from matplotlib import pyplot as plt
from unet import UNet
from torch.nn import functional as F
import argparse
import glob
import numpy as np
from skimage import io, img_as_ubyte, exposure
import os
import json

def save_to_json(data, filepath, mode='w'):
    with open(filepath, mode) as file:
        file.write(json.dumps(data))

def unet_padding_torch(npimage,n_down=4):
    n = 2**n_down
    shape = npimage.shape
    h_pad = n - shape[1]%n
    #if h_pad==n:
    #    h_pad = 0
    w_pad = n - shape[2]%n
    #if w_pad==n:
    #    w_pad = 0
    h_halfpad = int(h_pad/2)
    w_halfpad = int(w_pad/2)
    return (w_halfpad, w_pad-w_halfpad, h_halfpad, h_pad-h_halfpad)  

def save_npimage(img, file):
    io.imsave(file, img_as_ubyte(img)) #exposure.rescale_intensity(img,(0,1),(0,1)))
    
def save_npimage_fullrange(img, file):
    io.imsave(file, exposure.rescale_intensity(img,'image',(0,1)))

def save_image(img, file):
    io.imsave(file, (255*img).astype(np.uint8))

def to_numpy(torch_img):
    np_img = torch_img.numpy()
    if np_img.shape[0] == 1:
        np_img = np_img.reshape(np_img.shape[1:3])
    else:
        np_img = np_img.transpose(1,2,0)
    return np_img
    
def get_dice_from_jaccard(jaccard_score):
    return 2*jaccard_score/(1+jaccard_score)

def load_model(model, filepath, strict=True):
    model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())), strict=strict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--results_folder', type=str)
    parser.add_argument('--path_dataset', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='unet')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--compute_pr', action='store_true')
    args = parser.parse_args()

    DEVICE = 'cuda:'+str(args.device)
    DATA_FOLDER = args.path_dataset
    RESULTS_FOLDER = args.results_folder
    MODEL_TYPE = args.model_type
    MODEL_FILE = args.model_file
    SAVE_IMAGES = args.save_images
    COMPUTE_PR = args.compute_pr

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    #Find data
    image_list = list(glob.glob(DATA_FOLDER+'/*tif'))
    padding = unet_padding_torch

    #Create and load model
    #Modify this to load the desired network
    if MODEL_TYPE=='unet':
        network = UNet(3,1,64).to(DEVICE) 
    load_model(network, MODEL_FILE)
    network.eval()

    #Run model through data
    results = []

    count = 0

    print('Image processing')

    for image_file in image_list:
        print(image_file)
        id = image_file.split('/')[-1][0:2] #.split('.')[0]
        mask_file = DATA_FOLDER + '/' + id + '_test_mask.png'
        v_file = DATA_FOLDER + '/' + id + '_manual1.gif'

        img = Image.open(image_file)
        img = tr.functional.to_tensor(img)

        mask = np.asarray(Image.open(mask_file))
        mask = (mask/mask.max()).astype(int)

        v = np.asarray(Image.open(v_file))
        v = (v/v.max()).astype(int)

        print(v.min(), v.max())

        if padding is not None:
            print(img.shape)
            pad = padding(img)
            print(pad)
            img = F.pad(img, pad)
            print(img.shape)

        img = img.unsqueeze(0).to(DEVICE)

        # Forward pass through the network. 
        with torch.no_grad():
            # Modify according to the network architecture in order to save the output of task OD/OC
            prediction = network(img) #[:,4:7,:,:]
        
        prediction = F.sigmoid(prediction).squeeze(0).squeeze(0)[pad[2]:-pad[3],pad[0]:-pad[1]]

        if SAVE_IMAGES:
            save_npimage(prediction.data.cpu().numpy(), RESULTS_FOLDER + '/' + id + '_prediction.jpg')           

        results += [[v[mask>0], prediction[mask>0].cpu().numpy()]]

        count += 1

    #Compute metrics

    print('Metrics computation')

    aucroc = {}
    aucpr = {}
    # print(results)
    eval_metrics = {}
 
    target, pred = zip(*results)

    target = np.concatenate([t.ravel() for t in target], axis=0)       
    pred = np.concatenate([p.ravel() for p in pred], axis=0)
    
    fpr, tpr, thr = metrics.roc_curve(target, pred)
    eval_metrics['aucroc'] = metrics.auc(fpr, tpr)
            
    if COMPUTE_PR:               
        precision, recall, thr = metrics.precision_recall_curve(target, pred)
        eval_metrics['aucpr'] = metrics.auc(recall, precision)
        
    save_to_json(eval_metrics, RESULTS_FOLDER+'/results.json')

    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr, label='AUC=%0.4f' % eval_metrics['aucroc'])
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(loc="lower left")
    plt.savefig(RESULTS_FOLDER+'/roc_curve.png', format='png', bbox_inches='tight')
    plt.close()
    
    if COMPUTE_PR:           
        plt.figure()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision, label='AUC=%0.4f' % eval_metrics['aucpr'])
        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.legend(loc="lower left")
        plt.savefig(RESULTS_FOLDER+'/pr_curve.png', format='png', bbox_inches='tight')
        plt.close()

