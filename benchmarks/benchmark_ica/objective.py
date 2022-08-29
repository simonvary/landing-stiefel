import numpy as np

from picard import amari_distance

from benchopt import BaseObjective



def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(-2 * Y))

class Objective(BaseObjective):
    name = "ICA under orthogonal constraint"

    parameters = {
    }

    def __init__(self):
        pass

    def set_data(self, X, mixing):
        self.X = X
        self.mixing = mixing

    def compute(self, unmixing):
        Y = np.dot(self.X, unmixing.T)
        loss = np.mean(_logcosh(Y))
        amari = amari_distance(unmixing, self.mixing)
        n, p = unmixing.shape
        error = np.linalg.norm(np.dot(unmixing.T, unmixing) - np.eye(p))
        return {"value": loss, "Amari distance": amari, "orth error":  error}

    def to_dict(self):
        return dict(X=self.X, mixing=self.mixing)
