"""
    Benchmark for PCA
"""

import sys, os, random
import pickle
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


seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

filename = '1_pca.pt'

n_runs = 10

methods_labels = ['landing', 'retraction (QR)', 'regularization lam = 1', 'regularization lam = 1e3']

batch_size = 128
n_epochs = 60

def scheduler_function(optimizer):
    return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50], gamma=0.1))

# n_epochs3 = 80
# def scheduler_function_test3(optimizer):
#     return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70], gamma=0.1))

problems = {
    # 'test1': {
    #     'n_samples' : 15000,
    #     'n_features': 5000, 
    #     'p_subspace': 200, 
    #     'noise_sdev': 1e-1 
    # }, 
    'test2': {
       'n_samples' : 15000,
       'n_features': 5000,
       'p_subspace': 500,
       'noise_sdev': 1e-1,
    },
    # 'test3': {
    #    'n_samples' : 15000,
    #    'n_features': 5000,
    #    'p_subspace': 1000,
    #    'noise_sdev': 1e-1
    # },
    #'test4': {
    #    'n_samples' : 15000,
    #    'n_features': 5000,
    #    'p_subspace': 100,
    #    'noise_sdev': 1e-1
    # }
}

methods = {
    'landing1': {
        'method_name': 'landing',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-3,
        'lambda_regul': 10, 
        'safe_step': 0.5, 
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': torch.device('cuda')
    },
    'retraction1': {
        'method_name': 'retraction',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-3,
        'lambda_regul': 1, 
        'safe_step': 0.5, 
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': torch.device('cuda')
    },
    'regularization1': {
        'method_name': 'regularization',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-3,
        'lambda_regul': 1e2, 
        'safe_step': None, 
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': torch.device('cuda')
    },
    'regularization2': {
        'method_name': 'regularization',
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'learning_rate': 1e-5,
        'lambda_regul': 1e4, 
        'safe_step': None, 
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': torch.device('cuda')
    }
}
if not os.path.exists('outputs'):
    os.makedirs('outputs')

print('Starting for file ' + filename)

out = {}

for problem_id in problems:
    problem_params = problems[problem_id]
    out[problem_id] = {}
    print('Starting with problem: '+ problem_id)

    results = {}
    for method_id, method_label in zip(methods, methods_labels):
        method_params = methods[method_id]
        method_name = methods[method_id]['method_name']
        print("\tSolver: "+ method_id)
        out[problem_id][method_id] = {}
        for run_id in range(n_runs):
            print("\t\tRun: {:d}/{:d}".format(run_id+1,n_runs))
            out[problem_id][method_id][run_id] = run_pca_experiment(problem_params, method_name, method_params)
        
        # Setup numpy array of all the runs via reference 
        out_tmp = out[problem_id][method_id]
        out_tmp['train_loss'] = np.array(out_tmp[0]['train_loss'])
        out_tmp['stiefel_distances'] = np.array(out_tmp[0]['stiefel_distances'])
        out_tmp['time_list'] = np.array(out_tmp[0]['time_list'])
        for run_id in range(1, n_runs):
            out_tmp['train_loss'] = np.vstack((out_tmp['train_loss'], out_tmp[run_id]['train_loss'] ))
            out_tmp['stiefel_distances'] = np.vstack((out_tmp['stiefel_distances'], out_tmp[run_id]['stiefel_distances'] ))
            out_tmp['time_list'] = np.vstack((out_tmp['time_list'], out_tmp[run_id]['time_list']))
        results[method_id] = {
            'time_list' : out_tmp['time_list'],
            'stiefel_distances': out_tmp['stiefel_distances'],
            'train_loss' : out_tmp['train_loss']
        }
        torch.save({
        'out': out,
        'n_runs': n_runs,
        'methods_labels': methods_labels,
        'methods': methods,
        'problems': problems}, 'outputs/1_pca/'+filename)
    with open('outputs/1_pca/'+problem_id+'.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
