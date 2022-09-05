from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import warnings

    import numpy as np
    from picard import picard

    

class Solver(BaseSolver):
    name = 'picard'

    def set_objective(self, X, mixing):
        self.X = X

    def run(self, n_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K, W, _ = picard(self.X.T, max_iter=n_iter + 1, ortho=True,
                             extended=False, tol=0, whiten=False)
        self.unmixing = W 

    def get_result(self):
        return self.unmixing
