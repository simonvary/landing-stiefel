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
n = 100         # dimension of samples
p = 20          # number of principal components
A = torch.randn(m, p) @ torch.randn(p, n)
init_weights = torch.randn(n, p)

_, _, vh = torch.linalg.svd(A, full_matrices = False)
x_star = vh[:p,:].T

# Objective: min -(1/2) * \| AX \|^2_F
loss_star = -.5 * (torch.matmul(A, x_star) ** 2).sum() / m
loss_star = loss_star.item()


method_names = ["Landing", "Retraction"] # "Retraction",
methods =  [LandingStiefelSGD, RiemannianSGD] # RiemannianSGD

learning_rate = 1e-3 # 0.3

f, axes = plt.subplots(2, 1)
for method_name, method, n_epochs in zip(method_names, methods, [5000, 5000]):
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
        loss = -.5 * (torch.matmul(A, param) ** 2).sum() / m
        loss.backward()
        time_list.append(time() - t0)
        loss_list.append(loss.item() - loss_star)
        iterates.append(param.data.clone())
        optimizer.step()

    distance_list = []
    for matrix in iterates:
        d = (
            torch.norm(matrix.transpose(-1, -2).matmul(matrix) - torch.eye(p))
        )
        distance_list.append(d.item())
    axes[0].semilogy(time_list, distance_list, label=method_name)
    axes[1].semilogy(time_list, loss_list, label=method_name)

axes[0].set_xlabel("time (s.)")
axes[1].set_xlabel("time (s.)")
axes[0].set_ylabel("Orthogonality error")
axes[1].set_ylabel("f - f^*")
plt.legend()
plt.show()
