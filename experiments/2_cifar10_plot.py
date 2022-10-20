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

def scheduler_function_geotorch(optimizer):
    return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1))


foldername = 'outputs/2_cifar10_VGG16/'
methods_ids = ['landing1_new', 'retraction1', 'regularization1', 'regularization2', 'trivialization1']
methods_labels = ['landing', 'retraction (QR)', 'regularization lam = 1', 'regularization lam = 1e3', 'trivialization']
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

    out_method = {'time_list': [], 
                  'train_loss': [],
                  'test_loss': [],
                  'test_accuracy': [],
                  'stiefel_distances': []}
    
    for i in range(n_runs):
        filename = foldername+method_id+'_run'+ str(i) +'.pt'
        out_file = torch.load(filename)
        for key in out_method.keys():
            out_method[key].append(out_file[key])

    for key in list(out_method):
        out_method[key] = np.array(out_method[key])
        if key == "time_list":
            out_method[key] = out_method[key] - out_method[key].min()
            out_method[key] = out_method[key] / 60
        out_method[key + '_mean'] = out_method[key].mean(axis = 0)
        out_method[key + '_std'] = out_method[key].std(axis = 0)
        out_method[key + '_min'] = out_method[key].min(axis = 0)
        out_method[key + '_max'] = out_method[key].max(axis = 0)

    times_mean = out_method['time_list_mean']
    axs[0].plot(times_mean, out_method['train_loss_mean'], '-', label = method_label, color=colors[method_id])
    #axs[0].fill_between(times_mins_mean, train_loss_mean - sdev_rho*train_loss_std, train_loss_mean+sdev_rho*train_loss_std, alpha=0.3) # Standard deviation
    axs[0].fill_between(times_mean, out_method['train_loss_min'], out_method['train_loss_max'], alpha=0.3, color=colors[method_id])
    axs[1].semilogy(times_mean, out_method['stiefel_distances_mean'], '-', label = method_label, color=colors[method_id]) 
    #axs[1].fill_between(times_mins_mean, stiefel_distances_mean - sdev_rho*stiefel_distances_std, stiefel_distances_mean+sdev_rho*stiefel_distances_std, alpha=0.3) # Standard deviation area
    axs[1].fill_between(times_mean, out_method['stiefel_distances_min'], out_method['stiefel_distances_max'], alpha=0.3, color=colors[method_id])

    axs[2].plot(times_mean, out_method['test_accuracy_mean'], '-', label = method_label, color=colors[method_id])
    #axs[0].fill_between(times_mins_mean, train_loss_mean - sdev_rho*train_loss_std, train_loss_mean+sdev_rho*train_loss_std, alpha=0.3) # Standard deviation
    axs[2].fill_between(times_mean, out_method['test_accuracy_min'], out_method['test_accuracy_max'], alpha=0.3, color=colors[method_id])


axs[0].legend()
axs[1].legend()
axs[2].legend()

time_lim = 90 # 150 mins

axs[0].set_xlabel('time (min.)')
axs[0].set_ylabel('Train loss')
axs[0].set_xlim([0,time_lim])


axs[1].set_xlabel('time (min.)')
axs[1].set_ylabel('Stiefel distance')
axs[1].set_xlim([0,time_lim])
axs[1].set_ylim([1e-10,1])



axs[2].set_xlabel('time (min.)')
axs[2].set_ylabel('Test accuracy')
axs[2].set_ylim([0,100])
axs[2].set_xlim([0,time_lim])



fig.savefig("plot_cifar10.pdf", dpi=150)
fig.show()
