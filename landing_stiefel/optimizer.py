import numpy as np

import torch
import torch.optim.optimizer
import torch.nn.functional as f

import geoopt
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.optim.mixin import OptimMixin

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
    ):
        #if lr < 0.0:
        #    raise ValueError("Invalid learning rate: {}".format(lr))
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
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a positive momentum and zero dampening"
            )
        
        super().__init__(params, defaults, stabilize=stabilize)
        # for group in self.param_groups:
        #     if lambda_regul>0:
        #         for param in group["params"]:
        #             param = geoopt.ManifoldParameter(param.clone(),manifold=geoopt.Stiefel(canonical=False))
        #             with torch.no_grad():
        #                 param.proj_()
        # Create placeholder tensors for tracking prev states for BB stepsizes
        # (todo: reformulate to be easier to read)
        # for group in self.param_groups:
        #     if group["lr"] in ['BB1', 'BB2', 'ABB']:
        #         group["params_prev"] = []
        #         group["grads_prev"] = []
        #         for point in group['params']:
        #             group["params_prev"].append(point.detach().clone())
        #             group["grads_prev"].append(point.detach().clone())
        #             group["params_prev"][-1].requires_grad = False
        #             group["grads_prev"][-1].requires_grad = False
        #     if group["lr"] == 'WNGrad':
        #         group["bs"] = []
        #         for point in group['params']:
        #             group["bs"].append(0)

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
                if lambda_regul>0:
                    group["stiefel_distance"] = 0
                else:
                    group["stiefel_distance"] = None
                for point_ind in range(len(group["params"])):
                    point = group["params"][point_ind]
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
                            size_p = np.prod(point.shape[-2:])
                            rel_grad, normal_dir, distance_norm = _landing_direction(point.view(-1, size_p), grad.view(-1, size_p), lambda_regul)
                            rel_grad, normal_dir = rel_grad.view(point.shape), normal_dir.view(point.shape)
                        else:
                            raise RuntimeError( "Cannot enforce orthogonality on scalar weights."
                        )
                        group["stiefel_distance"] = np.sqrt(group["stiefel_distance"]**2 + distance_norm.item()**2)

                        if safe_step:
                            d = distance_norm
                            g = torch.linalg.norm(rel_grad + normal_dir)
                            max_step = _safe_step_size(d, g, lambda_regul, safe_step)
                            #print(d)
                            #print(g)
                            #max_step = max_step.flatten()[0]
                            #print(max_step)
                            #step_size_shape = list(point.shape)
                            #step_size_shape[-1] = 1
                            #step_size_shape[-2] = 1
                            learning_rate = torch.clip(max_step, max=learning_rate)#.view(*step_size_shape)
                            #print(learning_rate)
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
    
    def stiefel_distances(self):
        stiefel_distances = []
        for group in self.param_groups:
            stiefel_distances.append(group["stiefel_distance"])
        return stiefel_distances

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

                    # Apply momentum to the relative gradient
                    #if momentum > 0:
                    #    momentum_buffer = state["momentum_buffer"]
                    #    momentum_buffer.mul_(momentum).add_(
                    #        rel_grad, alpha=1 - dampening
                    #    )
                    #    if nesterov:
                    #        rel_grad = rel_grad.add_(momentum_buffer, alpha=momentum)
                    #    else:
                    #        rel_grad = momentum_buffer
                    
                    # If learning_rate is float do a fixed stepsize with safeguard
                    #if isinstance(learning_rate, float):
                    #    step_size = learning_rate  

                    # BB strategies as in Bin's paper
                    # elif learning_rate in ['BB1', 'BB2', 'ABB']:
                    #     if group["step"] == 1:
                    #         step_size = 0.01
                    #     else:
                    #         S = point - group["params_prev"][point_ind]
                    #         Y = (rel_grad + normal_dir) - group["grads_prev"][point_ind]
                    #         if (learning_rate == 'BB1') or (learning_rate == 'ABB' and group["step"] % 2 == 1):
                    #             a = torch.abs(S.view((1,-1)) @ Y.view((-1,1)))
                    #             b = S.norm()**2
                    #             step_size = b/a # because it is inverse stepsize in Bin's paper
                    #         elif (learning_rate == 'BB2') or (learning_rate == 'ABB' and group["step"] % 2 == 0):
                    #             a = Y.norm()**2
                    #             b = torch.abs(S.view((1,-1)) @ Y.view((-1,1)))
                    #             step_size = b/a
                    #     # Update tracking variables
                    #     group["params_prev"][point_ind].copy_(point)
                    #     group["grads_prev"][point_ind].copy_(rel_grad + normal_dir)
                    # elif learning_rate == 'WNGrad':
                    #     b = group["bs"][point_ind]
                    #     if group["step"] == 1:
                    #         group["bs"][point_ind] = torch.norm(rel_grad + normal_dir)
                    #     else:
                    #         group["bs"][point_ind] += torch.norm(rel_grad + normal_dir)**2 / group["bs"][point_ind]
                    #     step_size = 1/group["bs"][point_ind]
"""                    if safe_step:
                        d = distance_norm
                        g = torch.norm(rel_grad + normal_dir, dim=(-1, -2))
                        max_step = _safe_step_size(d, g, lambda_regul, safe_step)
                        #print(max_step)
                        # One step per orthogonal matrix
                        step_size_shape = list(point.shape)
                        step_size_shape[-1] = 1
                        step_size_shape[-2] = 1
                        step_size = torch.clip(max_step, max=step_size).view(*step_size_shape) """