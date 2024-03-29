import math

import numpy as np

import torch
import torch.optim.optimizer
import torch.nn.functional as f

import geoopt
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.optim.mixin import OptimMixin



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


__all__ = ["LandingStiefelSGD"]


def _check_orthogonal(param):
    if not hasattr(param, "manifold"):
        raise TypeError("Parameter should be a geoopt parameter")
    if not isinstance(
        param.manifold, geoopt.manifolds.stiefel.CanonicalStiefel
    ) and not isinstance(
        param.manifold, geoopt.manifolds.stiefel.EuclideanStiefel
    ):
        raise TypeError("Parameters should be on the Stiefel manifold")
    *_, p, q = param.shape
    if p < q:
        raise ValueError(
            "The second to last dimension of the parameters should be greater or equal than its last dimension. "
            "Only tall matrices are supported so far"
        )


def _safe_step_size(d, g, lambda_regul, eps_d):
    """Compute the safe step size

    Parameters
    ----------
    d : float
        The distance to the manifold
    g : float
        The norm of the landing update
    lambda_regul : float
        The regularisation parameter
    eps_d : float
        The tolerance: the maximal allowed distance to the manifold
    Return
    ------
    sol : float
        The maximal step-size one can take
    """
    beta = lambda_regul * d * (d-1)
    alpha = g**2
    sol = (- beta + torch.sqrt(beta**2 - alpha * (d - eps_d)))/alpha
    return sol


def _landing_direction(point, grad, lambda_regul):
    r'''
    Computes the relative gradient, the normal direction, and the distance 
    towards the manifold.
    '''

    *_, p = point.shape

    XtX = torch.matmul(point.transpose(-1, -2), point)
    GtX = torch.matmul(grad.transpose(-1, -2), point)
    distance = XtX - torch.eye(p, device=point.device)

    # Note, that if we didn't need to know the rel_grad and distance norm, 
    # this could be further sped up by doing point@(GtX-lam*distance)
    rel_grad = .5*(torch.matmul(grad, XtX) - torch.matmul(point, GtX))  
    norm_dir = lambda_regul * torch.matmul(point, distance)
    # Distance norm for _safe_step_size computation
    distance_norm = torch.norm(distance, dim=(-1, -2))

    return (rel_grad, norm_dir, distance_norm)


class LandingStiefelSGD(OptimMixin, torch.optim.Optimizer):
    r"""
    Landing algorithm on the Stiefel manifold with the same API as
    :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups. Must contain square orthogonal matrices.
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)
    lambda_regul : float (optional)
        the hyperparameter lambda that controls the tradeoff between
        optimization in f and landing speed (default: 1.)
    check_type : bool (optional)
        whether to check that the parameters are all orthogonal matrices

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stabilize=None,
        lambda_regul=0,
        normalize_columns=False,
        safe_step=0.5,
        check_type=False,
        use_vr=False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if lambda_regul < 0.0:
            raise ValueError(
                "Invalid lambda_regul value: {}".format(lambda_regul)
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            lambda_regul=lambda_regul,
            normalize_columns=normalize_columns,
            safe_step=safe_step,
            check_type=check_type,
            use_vr=use_vr
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a positive momentum and zero dampening"
            )
        
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                lambda_regul = group["lambda_regul"]
                normalize_columns = group["normalize_columns"]
                safe_step = group["safe_step"]
                check_type = group["check_type"]
                group["step"] += 1
                for point_ind, point in enumerate(group["params"]):
                    if check_type:
                        _check_orthogonal(point)
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "LandingSGD does not support sparse gradients."
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()
                    grad.add_(point, alpha=weight_decay)

                    # If orthogonalization is applied
                    if lambda_regul>0:
                        if len(point.shape) == 2:
                            rel_grad, normal_dir, distance_norm = _landing_direction(point, grad, lambda_regul)
                        elif len(point.shape) > 2:
                            point_ = reshape_conv2d_weight(point, split_ind = 3)
                            grad_ = reshape_conv2d_weight(grad, split_ind = 3)
                            rel_grad, normal_dir, distance_norm = _landing_direction(point_, grad_, lambda_regul)
                            rel_grad = reshape_conv2d_weight_back(rel_grad, point.shape, split_ind = 3)
                            normal_dir = reshape_conv2d_weight_back(normal_dir, point.shape, split_ind = 3)
                        else:
                            raise RuntimeError( "Cannot enforce orthogonality on scalar weights."
                        )
                        
                        if safe_step:
                            d = distance_norm
                            g = torch.linalg.norm(rel_grad + normal_dir)
                            max_step = _safe_step_size(d, g, lambda_regul, safe_step)
                            learning_rate = torch.clip(max_step, max=learning_rate)
                        
                        # Take the step with orthogonalization
                        new_point = point - learning_rate * (rel_grad + normal_dir)
                        if normalize_columns:
                            f.normalize(new_point, p=2, dim=-2, out=new_point)
                        if (group["stabilize"] is not None and 
                            group["step"] % group["stabilize"] == 0):
                            self.stabilize_group(group)
                    else: 
                        # Take the step without orthogonalization
                        new_point = point - learning_rate * grad
                    point.copy_(new_point)
        return loss
    

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            p.copy_(manifold.projx(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(manifold.proju(p, buf))





if __name__ == '__main__':
    m, p = 10, 5
    n = 200
    batch_size = 7
    data = torch.randn(n, m)
    x = torch.randn(m, p)
    x = torch.nn.Parameter(x)
    n_epochs = 3
    optim = LandingStiefelSGD(x, lr=0.1, lambda_regul=1.)
    optim.zero_grad()

    n_batch = math.ceil(n // batch_size)
    for i in range(n_epochs):
        randperm = np.random.permutation(n_batch)
        for idx in randperm:
            batch = slice(idx * batch_size, (idx + 1) * batch_size)
            x = data[slice]
            this_size, _ = x.shape
            loss = -torch.linalg.norm(data.mm(x)) ** 2 / this_size
            loss.backward()
            optim.step()
    