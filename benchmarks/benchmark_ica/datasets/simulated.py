from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.linalg import sqrtm


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (100000, 20),
            (10000, 10)
        ]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        sources = rng.laplace(size=(self.n_samples, self.n_features))
        mixing = rng.randn(self.n_features, self.n_features)
        X = np.dot(sources, mixing.T)
        W = np.linalg.pinv(sqrtm(X.T.dot(X) / self.n_samples))
        X = np.dot(X, W.T)
        mixing = np.dot(W, mixing)
        data = dict(X=X, mixing=mixing)

        return data
