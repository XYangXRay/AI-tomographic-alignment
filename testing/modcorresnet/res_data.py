"""
Module loads data for training
"""

import os
from os import listdir
import numpy as np
import dxchange
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# Normalizes projection data
def norm_data(proj):
    mean_tmp = np.mean(proj)
    std_tmp = np.std(proj)
    proj = (proj - mean_tmp) / std_tmp
    proj = (proj - proj.min()) / (proj.max() - proj.min())

# Determines if input is '.tiff' file type
def is_tiff_file(filename):
    return any(filename.endswith(extension) for extension in [".tiff"])

# Determiens if input is '.csv' file type
def is_csv_file(filename):
    return any(filename.endswith(extension) for extension in [".csv"])

# ???
def load_shu(filepath, h):
    data_1 = np.loadtxt(filepath[0], delimiter = ',')
    for i, file in enumerate(filepath[1:]):
        data_2 = np.loadtxt(file, delimiter = ',')
        data_1 = np.vstack((data_1, data_2))

    # Convert data to pytorch tensor
    data = torch.tensor(data_1).type(torch.FloatTensor)
    return data

# Loads projections
def load_proj(filepath):
    proj1 = dxchange.reader.read_tiff(filepath + '_1.tiff')
    proj1 = norm_data(proj1)[np.newaxis, ...]
    proj1 = torch.tensor(proj1).type(torch.FloatTensor)

    proj2 = dxchange.reader.read_tiff(filepath + '_2.tiff')
    proj2 = norm_data(proj2)[np.newaxis, ...]
    proj2 = torch.tensor(proj2).type(torch.FloatTensor)

    return proj1, proj2

# Class loads dataset from an input folder
class DatasetFromFolder(data.Dataset):

    def __init__(self, input_proj_dir, target_proj_dir):
        super(DatasetFromFolder, self).__init__()
        self.input_proj_filenames = []
        self.target_proj_filenames = []

        # Loads input projections from directory
        for name in listdir(input_proj_dir):
            self.tu = list(set(x.split('_')[0] for x in listdir(input_proj_dir + name) if is_tiff_file(x)))
            self.tu.sort()
            self.inputlen = len(self.tu)
            self.input_proj_filenames += [os.path.join(input_proj_dir + name, x) for x in self.tu]

        # Loads target projections from directory
        for name in listdir(target_proj_dir):
            shu = list(set(x.split('_')[0] for x in os.listdir(input_proj_dir + name) if is_csv_file(x)))
            self.target_proj_filenames += [os.path.join(target_proj_dir + name, x) for x in shu]
        
        self.target_labels = load_shu(self.target_proj_filenames, self.inputlen)

    # Returns fixed image, moving image, and target at index
    def __getitem__(self, index):
        input1, input2 = load_proj(self.input_proj_filenames[index])
        target = self.target_labels[index]
        return input1, input2, target

    # Returns length of input projections
    def __len__(self):
        return len(self.input_proj_filenames)