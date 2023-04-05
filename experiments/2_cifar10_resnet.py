import sys, os, random
from time import time
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import geoopt
import geotorch
from landing_stiefel import LandingStiefelSGD

from models import VGG16, ResNet18
from utils import stiefel_project, stiefel_distance, EuclideanStiefelConv2d, get_conv2d

from cifar10_experiment import run_cifar10_experiment

random_seed = 123
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

n_classes   = 10
batch_size  = 128
n_epochs    = 150
device      = torch.device('cuda')
n_runs = 5


model_name = "resnet18"
model = ResNet18
filename = 'outputs/2_cifar10_'+model_name+'.pt'
foldername = 'outputs/2_cifar10_'+model_name+'/'

if not os.path.exists(foldername):
    os.makedirs(foldername)


def scheduler_function(optimizer):
    return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1))

def scheduler_function_geotorch(optimizer):
    return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1))

def scheduler_function_geotorch2(optimizer):
    return(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Prepare the problem
problem = {}
problem['train_dataset'] = datasets.CIFAR10(
    root='../data', train=True, download=False, transform=transform_train)
problem['train_loader'] = torch.utils.data.DataLoader(
    problem['train_dataset'] , batch_size=batch_size, shuffle=True, num_workers=2)

problem['test_dataset'] = datasets.CIFAR10(
    root='../data', train=False, download=False, transform=transform_test)
problem['test_loader'] = torch.utils.data.DataLoader(
    problem['test_dataset'], batch_size=batch_size, shuffle=False, num_workers=2)

# Prepare methods
methods_labels = ['landing', 'retraction (QR)', 'regularization lam = 1', 'regularization lam = 1e3', 'trivialization']
methods = {
    'landing1': {
        'method_name': 'landing',
        'model': model,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'lambda_regul': 1,
        'safe_step': 0.5,
        'learning_rate': 1e-1,
        'weight_decay': 5e-4,
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': 'cuda'
    },
    'retraction1': {
        'method_name': 'retraction',
        'model': model,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'lambda_regul': 1,
        'safe_step': 0.5,
        'learning_rate': 1e-1,
        'weight_decay': 5e-4,
        'init_project': True,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': 'cuda'
    },
    'regularization1': {
        'method_name': 'regularization',
        'model': model,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'lambda_regul': 1,
        'safe_step': None,
        'learning_rate': 1e-1,
        'weight_decay': 5e-4,
        'init_project': False,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': 'cuda'
    },
    'regularization2': {
        'method_name': 'regularization',
        'model': model,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'lambda_regul': 1e3,
        'safe_step': None,
        'learning_rate': 1e-4,
        'weight_decay': 5e-4,
        'init_project': False,
        'scheduler' : scheduler_function,
        'x0': None,
        'device': 'cuda'
    },
    'trivialization1': {
        'method_name': 'trivialization',
        'model': model,
        'batch_size': batch_size,
        'n_epochs': 30,
        'lambda_regul': 1,
        'safe_step': None,
        'learning_rate': 1e-1,
        'weight_decay': 5e-4,
        'init_project': True,
        'scheduler' : scheduler_function_geotorch,
        'x0': None,
        'device': 'cuda'
    }
}

out = {}
print('Starting CIFAR10 experiment with '+model_name)

for method_id, method_label in zip(methods, methods_labels):
    method_params = methods[method_id]
    method_name = methods[method_id]['method_name']
    print("\tSolver: "+ method_id)
    out[method_id] = {}
    for run_id in range(n_runs):
        print("\t\tRun: {:d}/{:d}".format(run_id+1,n_runs))
        out[method_id][run_id] = run_cifar10_experiment(problem, method_name, methods[method_id],run_file_name=foldername+method_id+'_run'+str(run_id)+'.pt')

    torch.save({
        'out': out[method_id],
        'n_runs': n_runs,
        'method_label': method_label,
        'method': methods[method_id],
        'problem': problem}, foldername+method_id+'.pt')

    torch.save({
        'out': out,
        'n_runs': n_runs,
        'methods_labels': methods_labels,
        'methods': methods,
        'problem': problem}, filename)
