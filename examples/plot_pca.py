"""
A simple example of the Stiefel landing algorithm on PCA problem
"""
from time import time

import matplotlib.pyplot as plt

import torch
import geoopt
from geoopt.optim import RiemannianSGD

from landing_stiefel import LandingStiefelSGD

torch.manual_seed(1)

# generate random matrices

m = 1000        # number of samples
n = 100         # dimension
p = 40          # number of principal components
A = torch.randn(m, n)
init_weights = torch.randn(n, p)

_, _, vh = torch.linalg.svd(A, full_matrices = False)
x_star = vh[:p,:].T

# Objective: (1/m) * \| AX \|^2_F
loss_star = (torch.matmul(A, x_star) ** 2).sum() / m
loss_star = loss_star.item()


method_names = ["Landing", "Retraction"]
methods = [LandingStiefelSGD, RiemannianSGD]

learning_rate = 0.3

for method_name, method, n_epochs in zip(method_names, methods, [2000, 500]):
    iterates = []
    loss_list = []
    time_list = []

    param = geoopt.ManifoldParameter(
        init_weights.clone(), manifold=geoopt.Stiefel(canonical=False)
    )
    with torch.no_grad():
        param.proj_()
    optimizer = method((param,), lr=learning_rate)
    t0 = time()
    for _ in range(n_epochs):

        optimizer.zero_grad()
        loss = (torch.matmul(A, param) ** 2).sum() / m
        loss.backward()
        time_list.append(time() - t0)
        loss_list.append(loss.item() - loss_star)
        iterates.append(param.data.clone())
        optimizer.step()

    distance_list = []
    for matrix in iterates:
        d = (
            torch.norm(matrix.matmul(matrix.transpose(-1, -2)) - torch.eye(p))
            / n
        )
        distance_list.append(d.item())
    axes[0].semilogy(time_list, distance_list, label=method_name)
    axes[1].semilogy(time_list, loss_list, label=method_name)
