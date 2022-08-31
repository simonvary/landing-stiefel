import numpy as np

from picard import amari_distance

from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    PcaOracle = import_ctx.import_from(
        'helper_pca', 'PcaOracle'
    )


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))

class Objective(BaseObjective):
    name = "ICA under orthogonal constraint"

    parameters = {
    }

    def __init__(self):
        pass

    def set_data(self, X, n_sources):
        self.X = X
        self.n_sources = n_sources
        self.oracle = PcaOracle(X, n_sources)

    def compute(self, unmixing):
        loss = self.oracle.loss(unmixing)
        n, p = unmixing.shape
        error = np.linalg.norm(np.dot(unmixing.T, unmixing) - np.eye(p))
        return {"value": loss, "orth error":  error}

    def to_dict(self):
        return dict(X=self.X, n_sources=self.n_sources)
