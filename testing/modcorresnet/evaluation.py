"""
Module assesses accuracy of created model
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # ???

import numpy as np
import dxchange
import cv2

from res_data import DatasetFromFolder
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader