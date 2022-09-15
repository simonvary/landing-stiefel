"""
    Benchmark for PCA (plots)
"""

import sys
from time import time
sys.path.append("../")

import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch



filename = '1_pca2.pt'
out_file = torch.load(filename)

out = out_file['out']
n_runs = out_file['n_runs']
methods_labels = out_file['methods_labels']
methods = out_file['methods']
problems = out_file['problems']


sdev_rho = 5

for problem_id in problems:
    problem_params = problems[problem_id]
    print('Ploting: '+ problem_id)
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle('PCA '+problem_id)
    for method_id, method_label in zip(methods, methods_labels):
        method_params = methods[method_id]
        method_name = methods[method_id]['method_name']
        print("\t ploting: "+ method_id)
        out_tmp = out[problem_id][method_id]

        times_mins_mean = out_tmp['arr_time_list'].mean(axis=0) /60
        train_loss_mean = out_tmp['arr_train_loss'].mean(axis=0)
        train_loss_std = out_tmp['arr_train_loss'].std(axis=0)
        stiefel_distances_mean = out_tmp['arr_stiefel_distances'].mean(axis=0)
        stiefel_distances_std = out_tmp['arr_stiefel_distances'].std(axis=0)

        axs[0].semilogy(times_mins_mean, train_loss_mean, '-', label = method_label)
        axs[0].fill_between(times_mins_mean, train_loss_mean - sdev_rho*train_loss_std, train_loss_mean+sdev_rho*train_loss_std, alpha=0.3)
        axs[1].semilogy(times_mins_mean, stiefel_distances_mean, '-', label = method_label) 
        axs[1].fill_between(times_mins_mean, stiefel_distances_mean - sdev_rho*stiefel_distances_std, stiefel_distances_mean+sdev_rho*stiefel_distances_std, alpha=0.3)
        # , color=colors[method_name]
    
    axs[0].set_xlabel('time (minutes)')
    axs[0].set_ylabel('Train loss (objective)')
    axs[0].legend()
    axs[1].set_xlabel('time (minutes)')
    axs[1].set_ylabel('Stiefel distance (constraint)')
    axs[1].legend()
    #axs[1].set_xlim((0,1))
    #axs[0].set_xlim((0,1))
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(filename+"_"+problem_id+".pdf", dpi=150)
    plt.show()

