from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from picard import picard

class Solver(BaseSolver):
    name = 'picard'

    def set_objective(self, X, mixing):
        self.X = X

    def run(self, n_iter):
        K, W, _ = picard(self.X.T, max_iter=n_iter + 1, ortho=True,
                         extended=False, tol=0)
        self.unmixing = np.dot(W, K)

    def get_result(self):
        return self.unmixing
