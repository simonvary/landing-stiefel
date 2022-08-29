"""
A simple example of the Stiefel landing algorithm on PCA problem
"""
from time import time

import sys
sys.path.append("../")

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import geoopt

from geoopt.optim import RiemannianSGD, RiemannianLineSearch
from landing_stiefel import LandingStiefelSGD

torch.manual_seed(0)

n_classes   = 10
batch_size  = 128
epochs      = 150
device      = torch.device('cuda')


method_names = ['landing', 'regularization', 'retraction (QR)']
file_names = ['2_cifar10_landing_v2.pt', '2_cifar10_regularization_v2.pt', '2_cifar10_geoopt_v2.pt']


train_loss_values = {}
test_loss_values = {}
test_accuracy_values = {}
stiefel_distances = {}
time_list = {}

for method_name,filename in zip(method_names,file_names):
    checkpoint = torch.load(filename)
    train_loss_values[method_name] = checkpoint['train_loss_values']
    test_loss_values[method_name] = checkpoint['test_loss_values']
    test_accuracy_values[method_name] = checkpoint['test_accuracy_values']
    stiefel_distances[method_name] = checkpoint['stiefel_distances']
    time_list[method_name] = checkpoint['time_list']

colormap = plt.cm.Set1
colors = {}
for i in range(len(method_names)):
    colors[method_names[i]] = colormap.colors[i]

fig, axs = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle('CIFAR-10 Orthogonal VGG16')

# Loss subplot
for method_name in method_names:
    times_mins = np.array(time_list[method_name]) / 60
    axs[0].plot(times_mins, train_loss_values[method_name], '-', label = method_name, color=colors[method_name])
    axs[0].set_xlabel('time (minutes)')
    axs[0].set_ylabel('Train loss (objective)')
    #axs[0].set_xlim(0, 50)
    axs[0].legend()

# Test subplot
for method_name in method_names:
    times_mins = np.array(time_list[method_name]) / 60
    axs[1].plot(times_mins, test_accuracy_values[method_name], '-', label = method_name, color=colors[method_name])
    axs[1].set_xlabel('time (minutes)')
    axs[1].set_ylabel('Test accuracy')
    #axs[1].set_xlim(0, 50)
    axs[1].legend(loc="lower right")

# Distances subplot
for method_name in method_names:
    times_mins = np.array(time_list[method_name]) / 60
    if stiefel_distances[method_name]:
        axs[2].semilogy(times_mins, stiefel_distances[method_name], '--', label = method_name, color=colors[method_name])
    axs[2].set_xlabel('time (minutes)')
    axs[2].set_ylabel('Distance to the constraint')
    axs[2].legend()

fig.subplots_adjust(hspace=0.5)
plt.savefig("plot_cifar10.pdf", dpi=150)
plt.show()
