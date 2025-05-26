#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashvaro
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from matplotlib import pyplot as plt
from skimage import io, exposure, img_as_ubyte
import os
import numpy as np
import csv
from utils_pytorch import ReduceLROnPlateau, IsBest

from unet import UNet

from torch.optim import lr_scheduler




def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

def save_opt(opt, filepath):
    torch.save(opt.state_dict(), filepath)

def load_opt(opt, filepath):
    opt.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))


def save_npimage(img, file):
    io.imsave(file, img_as_ubyte(img))

def save_npimage_fullrange(img, file):
    io.imsave(file, exposure.rescale_intensity(img,'image',(0,1)))

def save_image(img, file):
    io.imsave(file, (255*img).astype(np.uint8))

def compose_rgb_npimages(npimages_list):
    img_list = list()
    for img in npimages_list:
        if len(img.shape) == 2:
            img_list.append(np.stack((img,img,img), axis=2))
        elif len(img.shape) == 3:
            img_list.append(img)
    return np.concatenate(img_list, axis=1)


def to_numpy(torch_img):
    np_img = torch_img.numpy()
    if np_img.shape[1] == 1:
        np_img = np_img.reshape(np_img.shape[2:4])
    else:
        np_img = np_img.reshape(np_img.shape[1:4])
        np_img = np_img.transpose(1,2,0)
    return np_img

def save_to_csv(data, filepath):
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(data)







def learning_curves(training, validation, outfile):
    plt.rcParams["figure.figsize"] = [16,9]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)  # create figure & 1 axis
    x, y1 = zip(*training)
    ax1.plot(x, y1, 'b', label='training')

    x, y1 = zip(*validation)
    ax1.plot(x, y1, 'r', label='validation')

    ax1.legend()
    ax1.set_yscale('log')

    fig.savefig(outfile)   # save the figure to file
    plt.close(fig)    # close the figure





class R2Vessels:
    def __init__(self, config, n=64):

        self.set_cuda_device(config.gpu_id)

        self.loss = nn.BCEWithLogitsLoss(reduce=False)

        if config.pretrained_path is None:
            self.net = UNet(input_ch=3, output_ch=1, base_ch=n).to(self.device)
            self.net.initialize()
        else:
            pretrained = UNet(input_ch=3, output_ch=config.pre_output_chs, base_ch=n).to(self.device)
            load_model(pretrained, config.pretrained_path)
            pretrained.outconv = nn.Conv2d(64,1,1,bias=True).to(self.device)
            init.kaiming_normal(pretrained.outconv.weight.data, a=0, mode='fan_out', nonlinearity='sigmoid')
            init.constant(pretrained.outconv.bias.data, 0)
            self.net = pretrained


    def set_cuda_device(self, gpu_id, no_cuda_ok=False):
        if torch.cuda.is_available():
            if gpu_id is None:
                self.device = torch.device('cuda', 0)
                torch.cuda.set_device(0)
            else:
                self.device = torch.device('cuda', gpu_id) 
                torch.cuda.set_device(gpu_id)
        else:
            if no_cuda_ok:
                self.device = torch.device('cpu')
            else:
                raise RuntimeError('CUDA is not available')


    def __call__(self, input_img):
        self.net.set_eval()
        return self.net(input_img)
    

    def train_iters(self, r2v_loader, number_iters):
        self.net.train()
        total_loss = 0.0

        len_r2v = len(r2v_loader)

        for k in range(number_iters):
            if self.iter%len_r2v == 0:
                self.r2v_iterator = iter(r2v_loader)

            _, data = next(self.r2v_iterator)

            retino, vessels, mask = (x.to(self.device, non_blocking=True) for x in data)

            self.optimizer.zero_grad()

            pred_vessels = self.net(retino)

            loss = torch.mean(self.loss(pred_vessels, vessels))
#            loss = torch.mean(self.loss(pred_vessels[mask>0], vessels[mask>0]))
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            self.iter += 1

        return [total_loss/number_iters]


    @torch.no_grad()
    def test(self, r2v_dataloader, prefix_to_save=None):
        self.net.eval()
        total_loss = 0.0

        len_r2v = len(r2v_dataloader)

        for _data in r2v_dataloader:
            k = _data[0].numpy()[0]
            data = _data[1]

            retino, vessels, mask = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True), \
            data[2].cuda(non_blocking=True)

            pred_vessels = self.net(retino)

            #print(k, _data[0])
            if prefix_to_save is not None:
                save_npimage(to_numpy((F.sigmoid(pred_vessels)).data.cpu()), prefix_to_save + str(k) + '.jpg')

            loss = torch.mean(self.loss(pred_vessels, vessels))
#            loss = torch.mean(self.loss(pred_vessels[mask>0], vessels[mask>0]))

            total_loss += loss.item()
    
        return [total_loss/len_r2v]


    def training(self, config, train_loader, valid_loader, output_folder):

        # Optimizer
        if config.optimizer == 'Adam':
                self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.init_lr, betas=(0.9,0.999)) #, weight_decay=self.config.weight_decay) #, weight_decay=1e-2)
        elif config.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=config.init_lr, momentum=0.9) #, weight_decay=self.config.weight_decay) #, weight_decay=1e-4)


        # Training schedule
        total_epochs = config.epochs
        lr_decays = config.lr_decays

        if total_epochs!=None:
            print('Training with a fixed number of epochs')
            print('Number of epochs: ', total_epochs)
            if lr_decays!=None:
                print('Learning rate decay milestones: ', lr_decays)
                scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_decays, gamma=0.1)
            else:
                print('No learning rate decay')
        else:
            print('Training with scheduler')
            print('Patience: ',config.patience)
            total_epochs = float('inf')
            scheduler = ReduceLROnPlateau(self.optimizer, output_folder, factor=0.1, patience=config.patience, \
                                      verbose=True, cooldown=0, threshold=1e-8, threshold_mode='abs', min_lr=config.min_lr, eps=1e-8)

        # Other training initializations
        os.makedirs(output_folder, exist_ok=True)

        save_period = config.patience
        valid_period = config.epoch_size

        save_to_csv([['iter','best_loss']], \
                     os.path.join(output_folder, 'best_loss.csv'))

        train_loss = list()
        valid_loss = list()
        all_train_loss = list()
        all_valid_loss = list()

        self.iter = 0
        epochs_count = 0
        training = True
        
        check_is_best = IsBest(lower_is_better=True)

        # First validation before training       
        prefix_to_save=None
        valid_loss.append([self.iter] + self.test(valid_loader, prefix_to_save))

        # Training loop
        while epochs_count<total_epochs and training is True:   
            epochs_count += 1

            # Training one epoch
            train_loss.append([self.iter+valid_period] + self.train_iters(train_loader, valid_period))

            # Check and prepare saving
            if epochs_count % save_period == 0:
                save = True
            else:
                save = False

            if save:
                save_path = os.path.join(output_folder, str(self.iter))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                prefix_to_save=save_path+'/'
            else:
                prefix_to_save=None

            # Validation
            valid_loss.append([self.iter] + self.test(valid_loader, prefix_to_save))

            # Check if best
            is_best = check_is_best(valid_loss[-1][1])                              
            if is_best:
                save_to_csv([[str(x) for x in valid_loss[-1]]], \
                             os.path.join(output_folder, 'best_loss.csv'))   
                save_model(self.net, output_folder + '/generator_best.pth')  

            # Run scheduler                
            if scheduler is not None:
                scheduler.step(valid_loss[-1][1], self.iter)
                training = scheduler.training()

            # Save
            if save:
                save_to_csv(train_loss, os.path.join(output_folder, 'train_loss.csv'))
                save_to_csv(valid_loss, os.path.join(output_folder, 'test_loss.csv'))
                all_train_loss += train_loss
                all_valid_loss += valid_loss
                train_loss = []
                valid_loss = []
                learning_curves(all_train_loss, all_valid_loss, output_folder + '/learning_curves.svg')
            
        # Final saving
        if len(train_loss)>0:
            save_to_csv(train_loss, os.path.join(output_folder, 'train_loss.csv'))
        if len(valid_loss)>0:
            save_to_csv(valid_loss, os.path.join(output_folder, 'test_loss.csv'))

        save_model(self.net, output_folder + '/generator_last.pth')
        #save_opt(self.optimizer, path_to_save + '/optimizer_last.pth')
        learning_curves(all_train_loss, all_valid_loss, output_folder + '/learning_curves.svg')

        return 
