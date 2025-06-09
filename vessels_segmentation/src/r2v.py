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

def save_to_csv(data, filepath, write_header=False):

    file_exists = os.path.exists(filepath)
    
    if write_header and not file_exists: # Write the header when creating the file
        with open(filepath, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(data[0])
            if len(data) > 1: # Rest of the data
                writer.writerows(data[1:])
    else: # Append data
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

def learning_curves_components(train_maxima, train_displacement, train_radius,
                             valid_maxima, valid_displacement, valid_radius, outfile):
    plt.rcParams["figure.figsize"] = [16,9]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)  # create figure & 1 axis
    
    x_train_max, y_train_max = zip(*train_maxima)
    x_train_disp, y_train_disp = zip(*train_displacement)
    x_train_rad, y_train_rad = zip(*train_radius)
    
    ax1.plot(x_train_max, y_train_max, 'b', label='maxima train')
    ax1.plot(x_train_disp, y_train_disp, 'g', label='displacement train')
    ax1.plot(x_train_rad, y_train_rad, 'c', label='radius train')

    x_valid_max, y_valid_max = zip(*valid_maxima)
    x_valid_disp, y_valid_disp = zip(*valid_displacement)
    x_valid_rad, y_valid_rad = zip(*valid_radius)
    
    ax1.plot(x_valid_max, y_valid_max, 'r', label='maxima valid')
    ax1.plot(x_valid_disp, y_valid_disp, 'y', label='displacement valid')
    ax1.plot(x_valid_rad, y_valid_rad, 'm', label='radius valid')

    ax1.legend()
    ax1.set_yscale('log')

    fig.savefig(outfile)   # save the figure to file
    plt.close(fig)    # close the figure
    
class R2Vessels:
    def __init__(self, config, n=64):
        self.set_cuda_device(config.gpu_id)

        self.loss = nn.BCEWithLogitsLoss(reduce=False) #Binary
        self.regression_loss = nn.MSELoss(reduce=False) #Regression

        if config.pretrained_path is None:
            self.net = UNet(input_ch=3, output_ch=4, base_ch=n).to(self.device)
            self.net.initialize()
        else:
            pretrained = UNet(input_ch=3, output_ch=config.pre_output_chs, base_ch=n).to(self.device)
            load_model(pretrained, config.pretrained_path)
            pretrained.outconv = nn.Conv2d(64,4,1,bias=True).to(self.device)
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
    
    def calculate_masked_mse(self, pred, target, mask):
        """
        Calculates MSE only at points where maxima exist (mask=1)
        Args:
            pred: network prediction
            target: ground truth
            mask: binary mask for maxima (1 where maximum exists, 0 elsewhere)
        Returns:
            Masked MSE loss averaged over valid points only
        """
        # Count number of valid points
        num_valid_points = torch.sum(mask)
        
        if num_valid_points > 0:
            # Calculate squared error only at maximum points
            squared_error = ((pred - target) ** 2) * mask
            return torch.sum(squared_error) / num_valid_points
        return torch.tensor(0.0).to(self.device)

    def train_iters(self, r2v_loader, number_iters):
        self.net.train()
        total_loss = 0.0
        total_maxima_loss = 0.0
        total_displacement_loss = 0.0
        total_radius_loss = 0.0

        len_r2v = len(r2v_loader)

        for k in range(number_iters):
            if self.iter%len_r2v == 0:
                self.r2v_iterator = iter(r2v_loader)

            _, data = next(self.r2v_iterator)

            vessels, mask, maxima, displacement, radius = (x.to(self.device, non_blocking=True) for x in data)

            self.optimizer.zero_grad()

            pred = self.net(vessels)
            
            # Each prediction in a different channel
            pred_maxima = pred[:,0:1,:,:]
            pred_displacement = pred[:,1:3,:,:]
            pred_radius = pred[:,3:4,:,:]

            # Binary loss for maxima using BCE
            maxima_loss = torch.mean(self.loss(pred_maxima, maxima))
            maxima_mask = (maxima > 0).float()
            displacement_loss = self.calculate_masked_mse(pred_displacement, displacement, maxima_mask)
            radius_loss = 3.0 * self.calculate_masked_mse(pred_radius, radius, maxima_mask)
            
            loss = maxima_loss + displacement_loss + radius_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_maxima_loss += maxima_loss.item()
            total_displacement_loss += displacement_loss.item()
            total_radius_loss += radius_loss.item()
            self.iter += 1

        # Return both total and individual losses
        return [total_loss/number_iters, 
                total_maxima_loss/number_iters,
                total_displacement_loss/number_iters,
                total_radius_loss/number_iters]


    @torch.no_grad()
    def test(self, r2v_dataloader, prefix_to_save=None):
        self.net.eval()
        total_loss = 0.0
        total_maxima_loss = 0.0
        total_displacement_loss = 0.0
        total_radius_loss = 0.0

        len_r2v = len(r2v_dataloader)

        for _data in r2v_dataloader:
            k = _data[0].numpy()[0]
            data = _data[1]

            vessels, mask, maxima, displacement, radius = (data[0].cuda(non_blocking=True), 
                                                         data[1].cuda(non_blocking=True),
                                                         data[2].cuda(non_blocking=True),
                                                         data[3].cuda(non_blocking=True),
                                                         data[4].cuda(non_blocking=True))

            pred = self.net(vessels)

            pred_maxima = pred[:,0:1,:,:]
            pred_displacement = pred[:,1:3,:,:]
            pred_radius = pred[:,3:4,:,:]

            # Binary loss for maxima using BCE
            maxima_loss = torch.mean(self.loss(pred_maxima, maxima))
            maxima_mask = (maxima > 0).float()
            displacement_loss = self.calculate_masked_mse(pred_displacement, displacement, maxima_mask)
            radius_loss = 3.0 * self.calculate_masked_mse(pred_radius, radius, maxima_mask)
            
            loss = maxima_loss + displacement_loss + radius_loss

            total_loss += loss.item()
            total_maxima_loss += maxima_loss.item()
            total_displacement_loss += displacement_loss.item()
            total_radius_loss += radius_loss.item()

        # Return both total and individual losses
        return [total_loss/len_r2v,
                total_maxima_loss/len_r2v,
                total_displacement_loss/len_r2v,
                total_radius_loss/len_r2v]


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
                     os.path.join(output_folder, 'best_loss.csv'), write_header=True)

        train_loss = list()
        valid_loss = list()
        train_maxima_loss = list()
        train_displacement_loss = list()
        train_radius_loss = list()
        valid_maxima_loss = list()
        valid_displacement_loss = list()
        valid_radius_loss = list()
        
        all_train_loss = list()
        all_valid_loss = list()
        all_train_maxima_loss = list()
        all_train_displacement_loss = list()
        all_train_radius_loss = list()
        all_valid_maxima_loss = list()
        all_valid_displacement_loss = list()
        all_valid_radius_loss = list()

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
            losses = self.train_iters(train_loader, valid_period)
            train_loss.append([self.iter+valid_period, losses[0]])  # Total loss
            train_maxima_loss.append([self.iter+valid_period, losses[1]])  # Maxima loss
            train_displacement_loss.append([self.iter+valid_period, losses[2]])  # Displacement loss
            train_radius_loss.append([self.iter+valid_period, losses[3]])  # Radius loss

            # Validation
            losses = self.test(valid_loader, prefix_to_save)
            valid_loss.append([self.iter, losses[0]])  # Total loss
            valid_maxima_loss.append([self.iter, losses[1]])  # Maxima loss
            valid_displacement_loss.append([self.iter, losses[2]])  # Displacement loss 
            valid_radius_loss.append([self.iter, losses[3]])  # Radius loss

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
                             os.path.join(output_folder, 'best_loss.csv'), write_header=False)   
                save_model(self.net, output_folder + '/generator_best.pth')  

            # Run scheduler                
            if scheduler is not None:
                scheduler.step(valid_loss[-1][1], self.iter)
                training = scheduler.training()

            # Save
            if save:
                save_to_csv(train_loss, os.path.join(output_folder, 'train_loss.csv'), write_header=False)
                save_to_csv(valid_loss, os.path.join(output_folder, 'test_loss.csv'), write_header=False)
                
                # Save component losses
                save_to_csv(train_maxima_loss, os.path.join(output_folder, 'train_maxima_loss.csv'), write_header=False)
                save_to_csv(train_displacement_loss, os.path.join(output_folder, 'train_displacement_loss.csv'), write_header=False)
                save_to_csv(train_radius_loss, os.path.join(output_folder, 'train_radius_loss.csv'), write_header=False)
                save_to_csv(valid_maxima_loss, os.path.join(output_folder, 'valid_maxima_loss.csv'), write_header=False)
                save_to_csv(valid_displacement_loss, os.path.join(output_folder, 'valid_displacement_loss.csv'), write_header=False)
                save_to_csv(valid_radius_loss, os.path.join(output_folder, 'valid_radius_loss.csv'), write_header=False)
                
                all_train_loss += train_loss
                all_valid_loss += valid_loss
                all_train_maxima_loss += train_maxima_loss
                all_train_displacement_loss += train_displacement_loss
                all_train_radius_loss += train_radius_loss
                all_valid_maxima_loss += valid_maxima_loss
                all_valid_displacement_loss += valid_displacement_loss
                all_valid_radius_loss += valid_radius_loss
                
                # Clear lists
                train_loss = []
                valid_loss = []
                train_maxima_loss = []
                train_displacement_loss = []
                train_radius_loss = []
                valid_maxima_loss = []
                valid_displacement_loss = []
                valid_radius_loss = []
                
                # Generate learning curves
                learning_curves(all_train_loss, all_valid_loss, output_folder + '/learning_curves.svg')
                learning_curves(all_train_loss, all_valid_loss, output_folder + '/learning_curves_total.svg')
                learning_curves_components(
                    all_train_maxima_loss, all_train_displacement_loss, all_train_radius_loss,
                    all_valid_maxima_loss, all_valid_displacement_loss, all_valid_radius_loss,
                    output_folder + '/learning_curves_components.svg')
    
        # Final saving
        if len(train_loss)>0:
            save_to_csv(train_loss, os.path.join(output_folder, 'train_loss.csv'), write_header=False)
            save_to_csv(train_maxima_loss, os.path.join(output_folder, 'train_maxima_loss.csv'), write_header=False)
            save_to_csv(train_displacement_loss, os.path.join(output_folder, 'train_displacement_loss.csv'), write_header=False)
            save_to_csv(train_radius_loss, os.path.join(output_folder, 'train_radius_loss.csv'), write_header=False)
            
        if len(valid_loss)>0:
            save_to_csv(valid_loss, os.path.join(output_folder, 'test_loss.csv'), write_header=False)
            save_to_csv(valid_maxima_loss, os.path.join(output_folder, 'valid_maxima_loss.csv'), write_header=False)
            save_to_csv(valid_displacement_loss, os.path.join(output_folder, 'valid_displacement_loss.csv'), write_header=False)
            save_to_csv(valid_radius_loss, os.path.join(output_folder, 'valid_radius_loss.csv'), write_header=False)

        save_model(self.net, output_folder + '/generator_last.pth')
        learning_curves(all_train_loss, all_valid_loss, output_folder + '/learning_curves_total.svg')
        learning_curves_components(
            all_train_maxima_loss, all_train_displacement_loss, all_train_radius_loss,
            all_valid_maxima_loss, all_valid_displacement_loss, all_valid_radius_loss,
            output_folder + '/learning_curves_components.svg')

        return
