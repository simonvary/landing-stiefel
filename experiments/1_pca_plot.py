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



filename = '1_pca.pt'
out_file = torch.load(filename)

out = out_file['out']
n_runs = out_file['n_runs']
methods_labels = out_file['methods_labels']
methods_ids = list(out_file['methods'].keys())
methods = out_file['methods']
problems = out_file['problems']
n_runs = out_file['n_runs']

sdev_rho = 5

results = {}

for problem_id in problems:
    problem_params = problems[problem_id]
    print('Ploting: '+ problem_id)

    colormap = plt.cm.Set1
    colors = {}
    for i in range(len(methods_ids)):
        colors[methods_ids[i]] = colormap.colors[i]
        
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle('PCA '+problem_id)
    for method_id, method_label in zip(methods, methods_labels):
        method_params = methods[method_id]
        method_name = methods[method_id]['method_name']
        print("\t ploting: "+ method_id)
        out_method = out[problem_id][method_id]

        out_method = {'time_list': [], 
                'train_loss': [],
                'test_loss': [],
                'stiefel_distances': []}

        selected_keys = ('arr_time_list', 'arr_train_loss', 'arr_train_loss', 'arr_stiefel_distances')

        for key in selected_keys:
            if key == "arr_time_list":
                out_method[key] = out_method[key] / 60
            out_method[key + '_mean'] = out_method[key].mean(axis = 0)
            out_method[key + '_std'] = out_method[key].std(axis = 0)
            out_method[key + '_min'] = out_method[key].min(axis = 0)
            out_method[key + '_max'] = out_method[key].max(axis = 0)

        times_mean = out_method['arr_time_list_mean']
        axs[0].semilogy(times_mean, out_method['arr_train_loss_mean'], '-', label = method_label, color=colors[method_id])

        axs[0].fill_between(times_mean, out_method['arr_train_loss_min'], out_method['arr_train_loss_max'], alpha=0.3, color=colors[method_id])
        axs[1].semilogy(times_mean, out_method['arr_stiefel_distances_mean'], '-', label = method_label, color=colors[method_id]) 
        axs[1].fill_between(times_mean, out_method['arr_stiefel_distances_min'], out_method['arr_stiefel_distances_max'], alpha=0.3, color=colors[method_id])
    
    axs[0].set_xlabel('time (minutes)')
    axs[0].set_ylabel('Train loss (objective)')
    axs[0].legend()
    axs[1].set_xlabel('time (minutes)')
    axs[1].set_ylabel('Stiefel distance (constraint)')
    axs[1].legend()
    #axs[1].set_ylim((0,None))
    axs[0].set_ylim((-20,None))
    axs[0].set_xlim((-0.01,1))
    axs[1].set_xlim((-0.01,1))
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(filename+"_"+problem_id+".pdf", dpi=150)
    plt.show()

