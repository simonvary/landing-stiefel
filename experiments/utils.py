'''
    Helper functions
'''

import numpy as np
import torch
import torch.nn as nn
import geoopt

__all__ = ["EuclideanStiefelConv2d", "stiefel_project", "stiefel_distance", "get_conv2d"]


class EuclideanStiefelConv2d(geoopt.manifolds.Stiefel):
    name = "Stiefel convolution(euclidean)"
    reversible = False

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        size_p = np.prod(x.shape[-2:])
        return u - (x.view(-1, size_p) @ geoopt.linalg.sym(x.view(-1, size_p).transpose(-1, -2) @ u.view(-1, size_p))).view(x.shape)

    egrad2rgrad = proju

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proju(y, v)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        return (u * v).sum()

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        size_p = np.prod(x.shape[-2:])
        q, r = geoopt.linalg.qr(x.view(-1,size_p) + u.view(-1,size_p))
        unflip = geoopt.linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q.view(x.shape)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        size_p = np.prod(x.shape[-2:])
        xtu = x.view(-1,size_p).transpose(-1, -2) @ u.view(-1,size_p)
        utu = u.view(-1,size_p).transpose(-1, -2) @ u.view(-1,size_p)
        eye = torch.zeros_like(utu)
        eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
        logw = geoopt.linalg.block_matrix(((xtu, -utu), (eye, xtu)))
        w = geoopt.linalg.expm(logw)
        z = torch.cat((geoopt.linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
        y = torch.cat((x.view(x.shape), u.view(x.shape)), dim=-1) @ w @ z
        return y


def stiefel_project(param):
    with torch.no_grad():
        U, _, V = torch.linalg.svd(param, full_matrices=False)
        return torch.einsum("...ik,...kj->...ij", U, V)

def stiefel_distance(parameter_list, device, requires_grad = False):
    distance = torch.tensor(0.,device=device,requires_grad = requires_grad)
    for param in parameter_list:
        if len(param.shape) == 4:
            size_p = np.prod(param.shape[-2:])
            distance += torch.norm(param.view(-1, size_p).T @ param.view(-1, size_p) - torch.eye(size_p,device=device))**2
        elif len(param.shape) == 2:
            size_p = param.shape[-1]
            distance += torch.norm(param.T @ param - torch.eye(size_p,device=device))**2
    return distance


def get_conv2d(model, project=False):
    module_list = nn.ModuleList()
    parameter_list = nn.ParameterList()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_list.append(module)
            if project:
                size_p = np.prod(module.weight.shape[-2:])
                module.weight = nn.Parameter(stiefel_project(module.weight.reshape(-1,size_p)).reshape(module.weight.shape))
            parameter_list.append(module.weight)

    other_parameter_list = nn.ParameterList()
    for param in model.parameters():
        if all(param is not o for o in parameter_list):
            other_parameter_list.append(param)

    return (module_list, parameter_list, other_parameter_list)