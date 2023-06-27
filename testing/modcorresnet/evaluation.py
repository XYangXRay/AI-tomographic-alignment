"""
Module assesses accuracy of created model
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # ???
import argparse
from argparse import ArgumentParser

import numpy as np
import dxchange
import cv2

from res_data import DatasetFromFolder
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import resnet18

# Normalizes projection data
def norm_data(proj):
    proj = (proj - torch.min(proj)) / (torch.max(proj) - torch.min(proj))
    return proj

# Determines if input is '.tiff' file type
def is_tiff_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff"])

# Determines if input is '.txt' file type
def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])

# Create np array of projections
def red_stack_tiff(path):
    files = os.listdir(path)
    proj = []

    for n, file in enumerate(files):
        if is_tiff_file(file):
            p = dxchange.read_tiff(path + file)
        proj.append(p)

    prj = np.array(proj)
    return prj

# Get arguments for evaluate.py from command line
def get_args():
    parser = argparse.ArgumentParser(description = 'Test the Neural Net', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test__path', type = str, default = '/path/to/data/predict/')
    parser.add_argument('--model_dir', type = str, default = '/path/to/data/test/')
    return parser.parse_args()

if __name__ == "__main__":
    opt = get_args()

    for index in indxs:
        test__path = opt.test__path

        for batch_size in range(182):
            print('batch_size : ', batch_size)
            test_set = DatasetFromFolder(test__path, test__path)
            testing_data_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = resnet18()
            model.to(device)

            plt_loss_max = []
            plt_loss = []
            plt_mean_loss = []

            ph_model = opt.model_dir
            model.load_state_dict(torch.load(ph_model))
            shifts = torch.tensor([])

            mse = nn.MSELoss(reduction = 'sum')
            mse.to(device)
            avg_loss_mse_proj = 0
            loss_shift = torch.tensor([])
            pred_shift = torch.tensor([])

            for input_fixed, input_moving, target in testing_data_loader:
                loss_shift, pred_shift = loss_shift.to(device), pred_shift.to(device)
                input_fixed, input_moving = norm_data(input_fixed), norm_data(input_moving)
                input_fixed, input_moving, target = input_fixed.to(device), input_moving.to(device), target.to(device)

                shift = model(input_fixed, input_moving)
                loss_mse_proj = mse(shift, target)
                avg_loss_mse_proj += loss_mse_proj
                
                a = shift - target
                loss_shift = torch.cat([loss_shift, a], dim = 0)
                pred_shift = torch.cat([pred_shift, shift], dim = 0)

                letest = np.ceil(len(testing_data_loader.dataset)) * 2.
                avg_loss_mse_proj = avg_loss_mse_proj / letest

                loss_shift = loss_shift.to('cpu').detach().numpy()
                pred_shift = pred_shift.to('cpu').detach().numpy()
                loss_shift_max = np.max(np.fabs(loss_shift))
                loss_shift_mean = np.mean(np.fabs(loss_shift))

                print('batch_size : ', batch_size,
                      "item = %s avg_loss_mse_proj = %f loss_max = %f , mean_loss = %f" % 
                      i, avg_loss_mse_proj.item, loss_shift_max, loss_shift_mean)
                plt_mean_loss.append(loss_shift_mean)
                plt_loss_max.append(loss_shift_max)
                plt_loss.append(avg_loss_mse_proj.item())
            
            item_loss = np.argmin(plt_loss)
            item_max_loss = np.argmin(plt_loss_max)
            item_mean = np.argmin(plt_mean_loss)

            print('item', item_loss, 'item_loss_max', plt_loss_max[item_loss], 'item_mse', plt_loss[item_loss],
                  'item_mean', plt_mean_loss[item_loss])