"""
    Benchmark for PCA
"""


import sys
from time import time
sys.path.append("../")

import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import geoopt
import geotorch
from landing_stiefel import LandingStiefelSGD

from utils import stiefel_project, stiefel_distance, generate_PCA_problem

from pca_experiment import run_pca_experiment




filename = '1_pca.pt'

n_runs = 5

methods_labels = ['landing', 'retraction (QR)', 'regularization lam = 1', 'regularization lam = 1e3']

batch_size = 128
n_epochs = 10

methods = {
    'landing1': {
        'method_name': 'landing',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-2,
        'lambda_regul': 1, 
        'safe_step': 0.5, 
        'init_project': True,
        'x0': None,
        'device': torch.device('cuda')
    },
    'retraction1': {
        'method_name': 'retraction',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-2,
        'lambda_regul': 1, 
        'safe_step': 0.5, 
        'init_project': True,
        'x0': None,
        'device': torch.device('cuda')
    },
    'regularization1': {
        'method_name': 'retraction',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-2,
        'lambda_regul': 1, 
        'safe_step': 0.5, 
        'init_project': True,
        'x0': None,
        'device': torch.device('cuda')
    },
    'regularization2': {
        'method_name': 'regularization',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-3,
        'lambda_regul': 1e2, 
        'safe_step': 0.5, 
        'init_project': True,
        'x0': None,
        'device': torch.device('cuda')
    }
}

problems = {
    'test1': {
        'n_samples' : 1000,
        'n_features': 200,
        'p_subspace': 100,
        'noise_sdev': 1e-1
    }, 
    'test2': {
        'n_samples' : 1000,
        'n_features': 200,
        'p_subspace': 100,
        'noise_sdev': 2*1e-2
    }
}


out = {}

for problem_id in problems:
    problem_params = problems[problem_id]
    out[problem_id] = {}
    print('Starting with problem: '+ problem_id)

    for method_id, method_label in zip(methods, methods_labels):
        method_params = methods[method_id]
        method_name = methods[method_id]['method_name']
        print("\tSolver: "+ method_id)
        out[problem_id][method_id] = {}
        for run_id in range(n_runs):
            out[problem_id][method_id][run_id] = run_pca_experiment(problem_params, method_name, method_params)
        
        # Setup numpy array of all the runs
        out_tmp = out[problem_id][method_id]
        out_tmp['arr_train_loss'] = np.array(out_tmp[0]['train_loss'])
        out_tmp['arr_stiefel_distances'] = np.array(out_tmp[0]['stiefel_distances'])
        out_tmp['arr_time_list'] = np.array(out_tmp[0]['time_list'])
        for run_id in range(1, n_runs):
            out_tmp['arr_train_loss'] = np.vstack((out_tmp['arr_train_loss'], out_tmp[run_id]['train_loss'] ))
            out_tmp['arr_stiefel_distances'] = np.vstack((out_tmp['arr_stiefel_distances'], out_tmp[run_id]['stiefel_distances'] ))
            out_tmp['arr_time_list'] = np.vstack((out_tmp['arr_time_list'], out_tmp[run_id]['time_list']))
        torch.save({
        'out': out,
        'n_runs': n_runs,
        'methods_labels': methods_labels,
        'methods': methods,
        'problems': problems}, filename)