import torch
import numpy as np
from typing import Union, Tuple, Optional

from geoopt.manifolds import Stiefel
from geoopt import linalg


__all__ = ["EuclideanStiefelConv2d"]


_stiefel_doc = r"""
    Manifold induced by the following matrix constraint:
        Stiefel convolution(euclidean)
    .. math::

        X^\top X = I\\
        X \in \mathrm{R}^{n\times m}\\
        n \ge m
"""


class EuclideanStiefelConv2d(Stiefel):
    name = "Stiefel convolution(euclidean)"
    reversible = False

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        size_p = np.prod(x.shape[-2:])
        return u - (x.view(-1, size_p) @ linalg.sym(x.view(-1, size_p).transpose(-1, -2) @ u.view(-1, size_p))).view(x.shape)

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
        q, r = linalg.qr(x.view(-1,size_p) + u.view(-1,size_p))
        unflip = linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q.view(x.shape)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        size_p = np.prod(x.shape[-2:])
        xtu = x.view(-1,size_p).transpose(-1, -2) @ u.view(-1,size_p)
        utu = u.view(-1,size_p).transpose(-1, -2) @ u.view(-1,size_p)
        eye = torch.zeros_like(utu)
        eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
        logw = linalg.block_matrix(((xtu, -utu), (eye, xtu)))
        w = linalg.expm(logw)
        z = torch.cat((linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
        y = torch.cat((x.view(x.shape), u.view(x.shape)), dim=-1) @ w @ z
        return y