#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ashvaro
"""
import matplotlib
#Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')


from r2v import R2Vessels
from data import VesselsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import random
from utils_dataset import read_samples_idx, save_samples_experiments, random_affine, random_hsv, read_samples_experiments, random_horizontal_flip, random_vertical_flip, to_torch_tensors
from torch.utils.data.sampler import Sampler
import os
import csv
import itertools
import time
import numpy as np
import argparse

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


def save_to_csv(data, filepath):
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        

def create_dataloaders(Dataset, datapath, train_idx, test_idx, transforms_train, transforms_test, target_pattern=None, orig_pattern=None, mask_pattern=None):
    dataset_train = Dataset(datapath, target_pattern, orig_pattern, mask_pattern, transforms.Compose(transforms_train)) 
    dataset_test = Dataset(datapath, target_pattern, orig_pattern, mask_pattern, transforms.Compose(transforms_test))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetSequentialSampler(test_idx)

    train_loader = DataLoader(dataset_train, batch_size=1, sampler=train_sampler, num_workers=1, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=1, drop_last=False, pin_memory=True, worker_init_fn=seed_worker)  
    
    return train_loader, test_loader, dataset_test



def train(training_path, train_idx, test_idx):

    r2a_train_loader, r2a_test_loader, dataset = create_dataloaders(Dataset=VesselsDataset,
                                                   datapath=config.path_dataset,
                                                   train_idx=list(train_idx),
                                                   test_idx=list(test_idx),
                                                   transforms_train=[VesselsDataset.drive_padding,
                                                                     random_affine,
                                                                     random_hsv,
                                                                     random_vertical_flip,
                                                                     random_horizontal_flip,
                                                                     to_torch_tensors],
                                                    transforms_test=[VesselsDataset.drive_padding, to_torch_tensors],
                                                    target_pattern='[0-9]+_manual1[.]gif',
                                                    orig_pattern='[0-9]+_training[.]tif',
                                                    mask_pattern='[0-9]+_test_mask[.]png')    
     
    multimodel = R2Vessels(config)
    
    multimodel.training(config, train_loader=r2a_train_loader,
                       valid_loader=r2a_test_loader,
                       output_folder=training_path)



def training_n_samples(data_path, experiment_path, experiment_number, experiments_file):
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)    
        
    save_to_csv([['n','samples','elapsed_time']], os.path.join(experiment_path, 'times.csv'))
    
    samples = set(read_samples_idx(os.path.join(data_path, 'training.csv')))

    experiments = read_samples_experiments(experiments_file)
    
    print(experiments)
    for i in experiment_number:
        training_set = experiments[i]
        save_samples_experiments([[i,training_set]],os.path.join(experiment_path, 'training_set.csv'))
        
        i_path = experiment_path + '/' + str(i)
        
        if not os.path.exists(i_path):
            os.makedirs(i_path)
            
        training_set = set(training_set)
        validation_set = samples - training_set
            
        print(samples, training_set, validation_set)

        start_time = time.time()  
        
        train(i_path, training_set, validation_set)
        
        elapsed_time = time.time() - start_time
        
        to_save = [[i,training_set, elapsed_time]]
        
        save_to_csv(to_save, os.path.join(experiment_path, 'times.csv'))
         


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute the training of a transfer learning approach for vessels segmentation')
    parser.add_argument('--path_dataset', type=str, required=True, help='As its name implies, this is the path of the input dataset')
    parser.add_argument('--main_path', type=str, required=True, help='This is the main directory where the results of the training process will be stored')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path of the pretrained model to be used for transfer learning')
    parser.add_argument('--pre_output_chs', type=int, default=1, help='Number of output channels of the pretrained model')
    parser.add_argument('--results_path', type=str, required=True, help='Directory where the results will be stored (this path is appended to "main_path", SO IT MUST START WITH /)')
    parser.add_argument('--seeds_list', type=int, required=True, nargs='+', help='List of seeds for the splitting of the dataset')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file where the indexes of the used image for a experiment is stored (this path is appended to "main_path", SO IT MUST START WITH /)')
    parser.add_argument('--epochs', type=int, help='If specified, then the model is trained during a fixed number of epochs. If not, then an early stopping is used.')
    parser.add_argument('--lr_decays', type=int, nargs='+', help='If specified, the learning rate will decay in epochs specified in this list. If not, the normal scheduler will be used')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--epoch_size', type=int, default=20)
    parser.add_argument('--test_aucpr', action='store_true')

    config = parser.parse_args()

    print('==========================================================================================================')
    print('Arguments dictionary')
    print('==========================================================================================================')
    print(config.__dict__)


    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    training_n_samples(config.path_dataset, config.main_path + '/' + config.results_path, config.seeds_list, config.csv_file)
