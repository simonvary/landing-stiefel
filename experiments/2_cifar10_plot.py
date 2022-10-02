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

def scheduler_function(optimizer):
    return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1))

foldername = 'outputs/2_cifar10_VGG16/'
methods_ids = ['landing1', 'retraction1', 'regularization1', 'regularization2']
methods_labels = ['landing', 'retraction (QR)', 'regularization lam = 1', 'regularization lam = 1e3']
n_runs = 5

colormap = plt.cm.Set1
colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]

#sdev_rho = 3
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle('VGG16 CIFAR-10')
for method_id, method_label  in zip(methods_ids, methods_labels):
    print("\t ploting: "+ method_id)
    filename = foldername+method_id+'_new.pt'
    out_file = torch.load(filename)
    out = out_file['out'][method_id]

    times_mins_mean = out['arr_time_list'].mean(axis=0) /60
    train_loss_mean = out['arr_train_loss'].mean(axis=0)
    train_loss_std = out['arr_train_loss'].std(axis=0)
    train_loss_min = out['arr_train_loss'].min(axis=0)
    train_loss_max = out['arr_train_loss'].max(axis=0)

    test_accuracy_mean = out['arr_test_accuracy'].mean(axis=0)
    test_accuracy_min = out['arr_test_accuracy'].min(axis=0)
    test_accuracy_max = out['arr_test_accuracy'].max(axis=0)


    stiefel_distances_mean = out['arr_stiefel_distances'].mean(axis=0)
    stiefel_distances_std = out['arr_stiefel_distances'].std(axis=0)
    stiefel_distances_min = out['arr_stiefel_distances'].min(axis=0)
    stiefel_distances_max = out['arr_stiefel_distances'].max(axis=0)


    axs[0].plot(times_mins_mean, train_loss_mean, '-', label = method_label, color=colors[method_id])
    #axs[0].fill_between(times_mins_mean, train_loss_mean - sdev_rho*train_loss_std, train_loss_mean+sdev_rho*train_loss_std, alpha=0.3) # Standard deviation
    axs[0].fill_between(times_mins_mean, train_loss_min, train_loss_max, alpha=0.3, color=colors[method_id])
    axs[1].semilogy(times_mins_mean, stiefel_distances_mean, '-', label = method_label, color=colors[method_id]) 
    #axs[1].fill_between(times_mins_mean, stiefel_distances_mean - sdev_rho*stiefel_distances_std, stiefel_distances_mean+sdev_rho*stiefel_distances_std, alpha=0.3) # Standard deviation area
    axs[1].fill_between(times_mins_mean, stiefel_distances_min, stiefel_distances_max, alpha=0.3, color=colors[method_id])

    axs[2].plot(times_mins_mean, test_accuracy_mean, '-', label = method_label, color=colors[method_id])
    #axs[0].fill_between(times_mins_mean, train_loss_mean - sdev_rho*train_loss_std, train_loss_mean+sdev_rho*train_loss_std, alpha=0.3) # Standard deviation
    axs[2].fill_between(times_mins_mean, test_accuracy_min, test_accuracy_max, alpha=0.3, color=colors[method_id])

axs[0].legend()
axs[1].legend()
axs[2].legend()

axs[0].set_xlabel('time (min.)')
axs[0].set_ylabel('Train loss')

axs[1].set_xlabel('time (min.)')
axs[1].set_ylabel('Stiefel distance')

axs[2].set_xlabel('time (min.)')
axs[2].set_ylabel('Test accuracy')
axs[2].set_ylim([0,100])


fig.savefig("plot_cifar10.pdf", dpi=150)
fig.show()
