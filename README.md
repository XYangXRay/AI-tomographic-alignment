# AI Tomographic Alignment Project

This repository contains all of the progress made toward automatic tomographic alignment using a deep convolutional neural network. It includes all attempts at creating an algorithm to determine alignment parameters, including all of the code used for synthesizing and preparing data to be run in the created models.

The current iteration of the workflow for determining alignment parameters using deep learning involves the usage of two-dimensional convolutional neural networks using the Residual Neural Network architecture to add additional convolutional layers. The data input into the neural network is a cross-correlation of the original projection data and a corresponding projection from reconstructing and artificially reprojecting the reconstructed object. Future efforts will involve using a Fourier cross-correlation to better isolate translational changes to improve the performance of the model
