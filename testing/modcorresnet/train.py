"""
Module trains model for image alignment
"""

import os
import warnings
import argparse
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import dxchange
import cv2
from tqdm import tqdm

from model import resnet18
from res_data import DatasetFromFolder

# Sets device to GPU if cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Normalizes data
def norm_data(proj):
    proj = (proj - torch.min(proj)) / (torch.max(proj) - torch.min(proj))
    return proj

# Argument parser to take arguments from command line
def get_args():
    parser = argparse.ArgumentParser(description = 'Train the Net ', formatter_class = argparse.ArgumentDefaultHelpFormatter)
    parser.add_argument('--train_dataroot', type = str, default = 'path/to/data/train/')
    parser.add_argument('--test_dataroot', type = str, default = 'path/to/data/test/')
    parser.add_argument('--train_batch_size', type = int, default = 8)
    parser.add_argument('--test_batch_size', type = int, default = 8)
    parser.add_argument('--num_steps', type = int, default = 44444)
    parser.add_argument('--display_step', type = int, default = 5)
    parser.add_argument('--learning_rate', type = float, default = 1e-4)
    parser.add_argument('--model_dir', type = str, default = 'path/to/your/model')
    parser.add_argument('--weights_init', type = str, default = 'False')
    parser.add_argument('--shuffle', type = str, default = 'True')
    return parser.parse_args()

def train():
    # Take arguments and sets device to GPU if cuda available
    opt = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load training data
    train__path = opt.train_dataroot
    train_set = DatasetFromFolder(train__path, train__path)
    training_data_loader = DataLoader(dataset = train_set, batch_size = opt.train_batch_size, shuffle = opt.shuffle)

    # Load testing data
    test__path = opt.test_dataroot
    test_set = DatasetFromFolder(test__path, test__path)
    testing_data_loader = DataLoader(dataset = test_set, batch_size = opt.test_batch_size, shuffle = False)
    
    # Sets or creates directory for output model
    model_dir = opt.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Move net to device, set optimizer and loss function
    G = resnet18()
    G.to(device)
    optimizer = Adam(G.parameters(), lr = opt.learning_rate)
    mse = nn.MSELoss(reduction = 'sum')
    mse.to(device)

    # Set up printing training data processing
    writer = SummaryWriter()
    i_test = 0

    # Iterate through the training data for 'num_steps'
    for i in range(opt.num_steps):
        avg_epoch_Gloss = 0

        # Iterates though every pair of fixed and moving projections and corresponding target alignment
        for input_fixed, input_moving, target in tqdm(training_data_loader):
            
            # Normalizing and moving data to GPU
            input_fixed, input_moving = norm_data(input_fixed), norm_data(input_moving)
            input_fixed.to(device)
            input_moving.to(device)
            target.to(device)

            # Resetting gradient, forward propogation
            optimizer.zero_grad()
            shift = G(input_fixed, input_moving)

            # Determining loss, backwards propogation and optimizing
            loss = mse(shift, target)
            G_loss = loss
            G_loss.backward()
            optimizer.step()
            avg_epoch_Gloss += G_loss

        # Print loss function over training process
        le = np.ceil(len(training_data_loader.dataset)) * 2
        avg_epoch_Gloss = avg_epoch_Gloss / le
        writer.add_scalar('train_loss_fn', avg_epoch_Gloss, i)
        print('n_iter%d,train_loss = %f' (i, avg_epoch_Gloss.item()))

        # Display model information after certain amount of 'display_step'
        if i % opt.display_step == 0:
            avg_loss_mse_projection = 0

            for input_fixed, input_moving, targer in training_data_loader:
                input_fixed = input_fixed.to(device)
                input_moving = input_moving.to(device)
                shift = G(input_fixed, input_moving)

            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(G.state_dict(), save_file_name)

            for input_fixed, input_moving, targer in testing_data_loader:
                input_fixed, input_moving = norm_data(input_fixed), norm_data(input_moving)
                input_fixed = input_fixed.to(device)
                input_moving = input_moving.to(device)
                target = target.to(device)
                shift = G(input_fixed, input_moving)

                loss_mse_projection = mse(shift, target)
                avg_loss_mse_projection += loss_mse_projection

        letest = np.ceil(len(testing_data_loader)) * 2
        avg_loss_mse_projection = avg_loss_mse_projection / letest
        print("test%d, test_losse = %f" % (i_test, avg_loss_mse_projection.item()))
        i_test += 1

    writer.close()

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category = DeprecationWarning)
    train()