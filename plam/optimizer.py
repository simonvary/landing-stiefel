import torch
import torch.optim.optimizer

import geoopt
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.optim.mixin import OptimMixin

__all__ = ["PLAMStiefelSGD"]


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


def _safe_step_size(d, a, lbda, eps):
    """Compute the safe step size

    Parameters
    ----------
    d : float
        The distance to the manifold
    a : float
        The norm of the relative gradient
    lbda : float
        The hyper-parameter lambda of the landing algorithm
    eps : float
        The tolerance: the maximal allowed distance to the manifold
    Return
    ------
    sol : float
        The maximal step-size one can take
    """
    alpha = 2 * (lbda * d - a * d - 2 * lbda * d)
    beta = a ** 2 + lbda ** 2 * d ** 3 + 2 * lbda * a * d ** 2 + a ** 2 * d
    sol = (alpha + torch.sqrt(alpha ** 2 + 4 * beta * (eps - d))) / 2 / beta
    return sol


def _plam_direction(point, grad, lambda_regul):
    r'''
    Computes the relative gradient, the normal direction, and the distance 
    towards the manifold.
    '''

    *_, p = point.shape

    XtX = torch.matmul(point.transpose(-1, -2), point)
    GtX = torch.matmul(grad.transpose(-1, -2), point)
    distance = XtX - torch.eye(p, device=point.device)

    plam_grad = grad - .5*torch.matmul(point, GtX + GtX.transpose(-1, -2))  
    norm_dir = lambda_regul * torch.matmul(point, distance)
    # Distance norm for _safe_step_size computation
    distance_norm = torch.norm(distance, dim=(-1, -2))

    return (plam_grad, norm_dir, distance_norm)


class PLAMStiefelSGD(OptimMixin, torch.optim.Optimizer):
    r"""
    PLAM field on the Stiefel manifold with the same API as
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
        lambda_regul=1.0,
        safe_step=[],
        check_type=True,
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
            safe_step=safe_step,
            check_type=check_type,
        )
        for param in params:
            with torch.no_grad():
                param.proj_()
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
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
                safe_step = group["safe_step"]
                check_type = group["check_type"]
                group["step"] += 1
                for point in group["params"]:
                    if check_type:
                        _check_orthogonal(point)
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "LandingSGD does not support sparse gradients"
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()

                    grad.add_(point, alpha=weight_decay)
                    rel_grad, normal_dir, distance_norm = _plam_direction(point, grad, lambda_regul)
                    # Apply momentum to the relative gradient
                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(
                            rel_grad, alpha=1 - dampening
                        )
                        if nesterov:
                            rel_grad = rel_grad.add_(momentum_buffer, alpha=momentum)
                        else:
                            rel_grad = momentum_buffer
                    if safe_step:
                        d = distance_norm
                        a = torch.norm(rel_grad, dim=(-1, -2))
                        max_step = _safe_step_size(d, a, lambda_regul, safe_step)
                        # One step per orthogonal matrix
                        step_size_shape = list(point.shape)
                        step_size_shape[-1] = 1
                        step_size_shape[-2] = 1
                        step_size = torch.clip(max_step, max=learning_rate).view(*step_size_shape)
                    else:
                        step_size = learning_rate
                    new_point = point - step_size * (rel_grad + normal_dir)
                    point.copy_(new_point)

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)
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
