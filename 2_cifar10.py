from time import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import geoopt
from geoopt.optim import RiemannianSGD, RiemannianLineSearch

import geotorch

import numpy as np

from landing_stiefel import LandingStiefelSGD

torch.manual_seed(0)

n_classes   = 10
batch_size  = 128
epochs      = 150
device      = torch.device('cuda')

learning_rate = 1
weight_decay = 5e-4
lambda_regul = 1
safe_step = None
method_name = 'geotorch'


class OrthConv2d(nn.Conv2d):
    def __init__(self, *args, alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        weight = self._parameters.pop("weight")
        self._weight_shape = weight.shape
        self.weight_orig = geoopt.ManifoldParameter(
            weight.data.reshape(weight.shape[0], -1), manifold=geoopt.Stiefel()
        )
        self.weight_orig.proj_()
        if alpha is None:
            self.alpha_orig = nn.Parameter(torch.zeros(self._weight_shape[0]))
        else:
            self.alpha_orig = alpha

    @property
    def weight(self):
        return (self.alpha[:, None] * self.weight_orig).reshape(self._weight_shape)

    @property
    def alpha(self):
        return self.alpha_orig.exp()


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, 10)
        self.loss_func =  nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def loss(self, logits, y):
        return self.loss_func(logits, y)

    def correct(self, logits, y):
        return torch.eq(torch.argmax(logits, dim=1), y) .sum()


def main():
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
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model and optimizers
    model = VGG16()
    model.to(device)

    if method_name == 'landing':
        # Parameter groups:
        ortho_params = []
        other_params = []
        for name, param in model.named_parameters():
            if len(param.shape) == 4:
                ortho_params.append(param)
            else:
                other_params.append(param)
        optimizer = LandingStiefelSGD([
                {'params': ortho_params, 'lambda_regul': lambda_regul, 'safe_step' : safe_step},
                {'params': other_params}], lr=learning_rate, weight_decay = weight_decay)
    elif method_name == 'geotorch':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                geotorch.orthogonal(module, 'weight')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    elif method_name == 'geoopt': # not working
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                size_p = np.prod(module.weight.shape[-2:])
                #module.weight = geoopt.ManifoldParameter(module.weight.view(-1, size_p), manifold=geoopt.Stiefel(canonical=False))
        optimizer = RiemannianSGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    elif method_name == 'regularization':
        def stiefel_regularization(model):
            stiefel_regul = torch.tensor(0.,device=device)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    size_p = np.prod(module.weight.shape[-2:])
                    weight_mat = module.weight.view(-1,size_p)
                    stiefel_regul += lambda_regul * torch.norm(weight_mat.T @ weight_mat - torch.eye(size_p,device=device))**2
            return stiefel_regul
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
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
                loss += stiefel_regularization(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch == 0:
            time_list.append(time() - time_start)
        else:
            time_list.append(time_list[-1] + (time() - time_start))
        train_loss_values.append(train_loss / len(train_loader))
        if method_name == 'landing':
            stiefel_distances.append(optimizer.stiefel_distances()[0])
            print("Epoch: {:d} Train set: Average loss: {:.4f} Distance: {:.4f}".format(epoch, train_loss_values[-1], stiefel_distances[-1]))
        elif method_name == 'regularization':
            dist_stiefel = np.sqrt(stiefel_regularization(model).item() / lambda_regul)
            stiefel_distances.append(dist_stiefel)
            print("Epoch: {:d} Train set: Average loss: {:.4f} Distance: {:.4f}".format(epoch, train_loss_values[-1], stiefel_distances[-1]))
        else:
            print("Epoch: {:d} Train set: Average loss: {:.4f}".format(epoch, train_loss_values[-1])) 

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
                }, '2_cifar10_'+method_name+'.pt')
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
            }, '2_cifar10_'+method_name+'.pt')
            
if __name__ == "__main__":
    main()