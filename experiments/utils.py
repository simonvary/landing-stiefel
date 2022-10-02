'''
    Helper functions
'''

import numpy as np
import torch
import torch.nn as nn
import geoopt
import geotorch

__all__ = ["EuclideanStiefelConv2d", "stiefel_project", "stiefel_distance", "get_conv2d",  "get_conv2d_weight_params", "reshape_conv2d_weight", "reshape_conv2d_weight_back", "orthogonalConv2d", "GeotorchStiefelConv2d"]


def reshape_conv2d_weight(x, split_ind = 3):
    size_n = np.prod(x.shape[:-split_ind])
    size_p = np.prod(x.shape[-split_ind:])
    x_ = x.view(size_n, size_p)
    if size_n >= size_p:
        return(x_)
    else:
        return(x_.T)

def reshape_conv2d_weight_back(x, shape, split_ind = 3):
    size_n = np.prod(shape[:-split_ind])
    size_p = np.prod(shape[-split_ind:])
    if size_n >= size_p:
        return (x.view(shape))
    else:
        return (x.T.view(shape))

class EuclideanStiefelConv2d(geoopt.manifolds.Stiefel):
    name = "Stiefel convolution(euclidean)"
    reversible = False

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_ = reshape_conv2d_weight(x, split_ind = 3)
        u_ = reshape_conv2d_weight(u, split_ind = 3)
        return u - reshape_conv2d_weight_back(x_ @ geoopt.linalg.sym(x_.transpose(-1, -2) @ u_), x.shape, split_ind = 3)

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
        x_ = reshape_conv2d_weight(x, split_ind = 3)
        u_ = reshape_conv2d_weight(u, split_ind = 3)
        q, r = geoopt.linalg.qr(x_ + u_)
        unflip = geoopt.linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return reshape_conv2d_weight_back(q, x.shape, split_ind=3)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_ = reshape_conv2d_weight(x, split_ind = 3)
        u_ = reshape_conv2d_weight(u, split_ind = 3)
        xtu = x_.transpose(-1, -2) @ u_
        utu = u_.transpose(-1, -2) @ u_
        eye = torch.zeros_like(utu)
        eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
        logw = geoopt.linalg.block_matrix(((xtu, -utu), (eye, xtu)))
        w = geoopt.linalg.expm(logw)
        z = torch.cat((geoopt.linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
        y = torch.cat((x_, u_), dim=-1) @ w @ z
        return y


def stiefel_project(param):
    with torch.no_grad():
        U, _, V = torch.linalg.svd(param, full_matrices=False)
        return torch.einsum("...ik,...kj->...ij", U, V)

def stiefel_distance(parameter_list, device, requires_grad = False):
    distance = torch.tensor(0.,device=device,requires_grad = requires_grad)
    for param in parameter_list:
        if len(param.shape) == 4:
            param_ = reshape_conv2d_weight(param, split_ind = 3)
            distance += torch.norm(param_.T @ param_ - torch.eye(param_.shape[-1],device=device))**2
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
                weight_ = reshape_conv2d_weight(module.weight, split_ind = 3)
                weight_ = stiefel_project(weight_)
                module.weight = nn.Parameter(reshape_conv2d_weight_back(weight_, module.weight.shape, split_ind = 3))
            parameter_list.append(module.weight)

    other_parameter_list = nn.ParameterList()
    for param in model.parameters():
        if all(param is not o for o in parameter_list):
            other_parameter_list.append(param)

    return (module_list, parameter_list, other_parameter_list)

def get_conv2d_weight_params(module_list):
    weight_list = []
    bias_list = []
    for module in module_list:
        weight_list.append(module.weight)
        bias_list.append(module.bias)
    return (weight_list, bias_list)

def size_flattened(size, dim):
    size = list(size)
    size_dim = size[dim]
    size[dim] = 1
    return (size_dim, np.prod(size))

class GeotorchStiefelConv2d(geotorch.Stiefel):
    def __init__(self, size, triv="expm"):
        super().__init__(size_flattened(size, 0), triv)
        self.size = list(size)
        self.transposed = self.size[0] < np.prod(self.size[1:])

    def _unfold(self, X):
        X = X.flatten(1)
        return X

    def _fold(self, X):
        return X.view(self.size)
    
    def forward(self, X):
        X = self._unfold(X)
        X = super().forward(X)
        X = self._fold(X)
        return X

    def right_inverse(self, X, check_in_manifold=True):
        if check_in_manifold and not self.in_manifold(X):
            raise geotorch.exceptions.InManifoldError(X, self)  
        X = self._unfold(X)
        X = super().right_inverse(X,check_in_manifold=False)
        return self._fold(X)
    
    def in_manifold(self, X, eps=1e-4):
        X = self._unfold(X)
        if X.size(-1) > X.size(-2):
            X = X.transpose(-2, -1)
        return geotorch.so._has_orthonormal_columns(X, eps)

    def sample(self, distribution="uniform", init_=None):
        X = super().sample(distribution, init_)
        X = self._fold(X)
        return X

def orthogonalConv2d(module, tensor_name="weight", triv="expm"):
    return geotorch.constraints._register_manifold(module, tensor_name, GeotorchStiefelConv2d, triv)


def generate_PCA_problem(m, n, p, sdev = 0):
    '''
    Generates A = L + N where L = U @ S @ V^T is a rank-r and 
    N is Gaussian noise
    '''
    U, _ = torch.linalg.qr(torch.randn(m,p), mode='reduced')
    V, _ = torch.linalg.qr(torch.randn(n,p), mode='reduced')
    S = torch.linspace(1, 0.5, steps=p)
    L = U @ S.diag() @ V.transpose(-1, -2)
    L = (L - L.mean(dim=0)) / L.std(dim=0) # zero mean, st. dev. one
    A = L + sdev * torch.randn(m,n)
    return A

