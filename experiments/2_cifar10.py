import sys
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

from models import VGG16
from utils import stiefel_project, stiefel_distance, EuclideanStiefelConv2d, get_conv2d

torch.manual_seed(0)

n_classes   = 10
batch_size  = 128
epochs      = 150
device      = torch.device('cuda')

learning_rate = 1e-1
weight_decay = 5e-4
lambda_regul = 1
safe_step = None
init_project = False
method_name = 'regularization'
filename = '2_cifar10_'+method_name+'_v2.pt'

if __name__ == "__main__":
    print('==> Preparing data..')
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

    train_dataset = datasets.CIFAR10(
        root='../data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(
        root='../data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model and optimizers
    model = VGG16()
    model.to(device)

    print('Method name: '+ method_name)
    print('File name: '+ filename)

    conv2d_modules, ortho_params, other_params = get_conv2d(model,project=init_project)
    print("Init. Stiefel distance: {:3.4e}".format(stiefel_distance(ortho_params,device = device).item()))

    if method_name == 'landing':
        optimizer = LandingStiefelSGD([
                {'params': ortho_params, 'lambda_regul': lambda_regul, 'safe_step' : safe_step},
                {'params': other_params}], lr=learning_rate, weight_decay = weight_decay)
    elif method_name == 'geotorch':
        for module in conv2d_modules:
            geotorch.orthogonal(module, 'weight')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    elif method_name == 'geoopt':
        for module in conv2d_modules:
            module.weight = geoopt.ManifoldParameter(module.weight, manifold=EuclideanStiefelConv2d())
        optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    if method_name == 'regularization':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1)

    # Train
    train_loss_values = []
    test_loss_values = []
    test_accuracy_values = []
    time_list = []
    stiefel_distances = []
    best_test_acc = 0.
    t0 = time()
    for epoch in range(epochs):
        time_start = time()
        model.train()
        train_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = model.loss(logits, batch_y)
            train_loss =+ loss.item() * batch_x.size(0)
            if method_name == 'regularization':
                loss += lambda_regul * stiefel_distance(ortho_params, device=device, requires_grad=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch == 0:
            time_list.append(time() - time_start)
        else:
            time_list.append(time_list[-1] + (time() - time_start))
        
        train_loss_values.append(train_loss / len(train_loader))
        stiefel_distances.append(stiefel_distance(ortho_params, device).item())

        print("Epoch: {:d} Train set: Average loss: {:.4f} Distance: {:3.4e}".format(epoch, train_loss_values[-1], stiefel_distances[-1]))

        # Test
        model.eval()
        with torch.no_grad():
            test_loss = 0.
            correct = 0.
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = model.loss(logits, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                correct += model.correct(logits, batch_y).item()

        test_loss_values.append(test_loss / len(test_dataset))
        test_accuracy_values.append(100 * correct / len(test_dataset))
        best_test_acc = max(test_accuracy_values[-1], best_test_acc)
        print("Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Best Accuracy: {:.2f}%".format(test_loss_values[-1], test_accuracy_values[-1], best_test_acc))

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_values': train_loss_values,
                'test_loss_values': test_loss_values,
                'test_accuracy_values': test_accuracy_values,
                'time_list': time_list,
                'stiefel_distances': stiefel_distances,
                }, filename)
        scheduler.step()
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_values': train_loss_values,
            'test_loss_values': test_loss_values,
            'test_accuracy_values': test_accuracy_values,
            'time_list': time_list,
            'stiefel_distances': stiefel_distances,
            }, filename)
