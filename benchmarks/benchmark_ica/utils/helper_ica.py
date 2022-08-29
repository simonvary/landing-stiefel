import numpy as np


def gradient(unmixing, X):
    Y = np.dot(X, unmixing.T)
    n, p = Y.shape
    return np.dot(np.tanh(Y.T), X) / n


def _logcosh(X):
    Y = np.abs(X)
    return Y + np.log1p(np.exp(- 2 * Y))


class IcaOracle(object):
    def __init__(self, X):
        self.X = X
        self.n_features = X.shape[1]

    def grad(self, x, slice):
        X_slice = self.X[slice]
        Y = np.dot(X_slice, x.reshape(self.n_features, self.n_features).T)
        n, _ = Y.shape
        # return np.dot(np.tanh(Y.T), X_slice) / n
        return np.dot(np.tanh(Y.T), Y) / n

    def loss(self, x, slice):
        X_slice = self.X[slice]
        Y = np.dot(X_slice, x.reshape(self.n_features, self.n_features).T)
        return np.mean(_logcosh(Y))