"""
A simple example of the Stiefel landing algorithm on PCA problem
"""
from time import time

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


method_names = ['landing', 'regularization', 'geotorch']

train_loss_values = {}
test_loss_values = {}
test_accuracy_values = {}
stiefel_distances = {}
time_list = {}
for method_name in method_names:
    checkpoint = torch.load('2_cifar10_'+method_name+'.pt')
    train_loss_values[method_name] = checkpoint['train_loss_values']
    test_loss_values[method_name] = checkpoint['test_loss_values']
    test_accuracy_values[method_name] = checkpoint['test_accuracy_values']
    stiefel_distances[method_name] = checkpoint['stiefel_distances']
    time_list[method_name] = checkpoint['time_list']

colors = {'landing' : 'blue', 
            'regularization' : 'red', 
            'geotorch' : 'green'}
for method_name in method_names:
    times_mins = np.array(time_list[method_name]) / 60
    plt.semilogy(times_mins, train_loss_values[method_name], '-', label = method_name + ' loss', color=colors[method_name])
    if stiefel_distances[method_name]:
        plt.semilogy(times_mins, stiefel_distances[method_name], '--', label = method_name + ' distance', color=colors[method_name])

plt.legend()
plt.savefig("plot_cifar10.pdf", dpi=150)
plt.show()