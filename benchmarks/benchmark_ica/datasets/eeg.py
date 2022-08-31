# from https://github.com/pierreablin/picard/blob/master/examples/plot_ica_eeg.py
from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.decomposition import PCA
    import mne
    from mne.datasets import sample


class Dataset(BaseDataset):

    name = "EEG"

    parameters = {
        'n_samples, n_features': [
            (20850, 10),
            (20850, 5)
        ]
    }

    def __init__(self, n_samples, n_features=30, random_state=42):
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        raw.filter(1, 40, n_jobs=1)  # 1Hz high pass is often helpful for fitting ICA

        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                            stim=False, exclude='bads')

        data = raw[picks, :][0]
        data = data[:, ::2]
        X = data.T
        pca = PCA(self.n_features, whiten=True)
        X = pca.fit_transform(X)
        transf = np.linalg.svd(rng.randn(self.n_features, self.n_features))[0]
        X = X.dot(transf.T)
        data = dict(X=X, mixing=np.eye(self.n_features))
        return self.n_features, data
