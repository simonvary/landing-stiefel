from random import random
import numpy as np

def landing_algorithm(fun_and_grad, x0, random_idxs, n_batch,
                      step_size, lbda, store_every, use_vr=True):
    n, p = x0.shape
    x = x0.copy()
    x_list = []
    n_iters = len(random_idxs)
    if use_vr:
        grad_memory = np.zeros((n_batch, n, p))
        for batch in range(n_batch):
            f_val, grad = fun_and_grad(x, batch)
            grad_memory[batch] = grad
        avg_gradient = np.mean(grad_memory, axis=0)
    for i in range(n_iters):
        if i % store_every == 0:
            x_list.append(x.copy())
        batch = random_idxs[i]
        f_val, grad = fun_and_grad(x, batch)
        if use_vr:
            old_grad = grad_memory[batch]
            direction = grad - old_grad + avg_gradient
            # direction = avg_gradient
            # update memory
            avg_gradient += (grad - old_grad) / n_batch
            grad_memory[batch] = grad
            
        else:
            direction = grad
        # Landing direction
        delta = np.dot(x.T, x)
        U = np.dot(direction.T, x)
        landing_direction = .5 * np.dot(direction, delta)
        landing_direction += np.dot(x, lbda * (delta - np.eye(p)) - .5 * U)
        x -= step_size * landing_direction
    _, g = fun_and_grad(x, batch, full=True)
    return x_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 10
    p = 5
    n_samples = 10000
    n_batch = 100   
    batch_size = n_samples // n_batch
    X = np.random.randn(n_samples, n)
    def fun_and_grad(x, batch, full=False):
        if full:
            data = X
        else:
            data = X[batch * batch_size: (batch + 1) * batch_size]
        y = data.dot(x)
        return None, -data.T.dot(y) / data.shape[0]

    def full_f(x):
        y = np.dot(X, x)
        return -np.mean(y ** 2)

    def orth_error(x):
        res = x.T.dot(x) - np.eye(p)
        return np.mean(res ** 2)

    def g_norm(x):
        _, g = fun_and_grad(x, 0, True)
        return np.linalg.norm(g - np.dot(x, np.dot(g.T, x)))

    def proj(x):
        return np.linalg.qr(x)[0]

    n_iters = 10000
    random_idxs = np.random.randint(0, n_batch, n_iters)
    step = 1e-1
    x0 = np.linalg.svd(np.random.randn(n, p), full_matrices=False)[0]
    plt.figure()
    for use_vr in [True, False]:
        x_list = landing_algorithm(fun_and_grad, x0, random_idxs, n_batch, step, 1., store_every=10, use_vr=use_vr)
        plt.plot([full_f(proj(x)) for x in x_list])
    plt.show()