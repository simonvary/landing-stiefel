# Landing algorithm on the Stiefel manifold


## PCA experiment

You can run the experiment with online PCA with `python 1_pca.py` in the folder `experiments/`, which will produce output files in the folder `experiments/outputs`.

To run a single instance of the experiment, run `python pca_experiment.py` or see the notebook `example_pca.ipynb`.

## ICA experiment

To run the ICA experiment, you need to install benchopt (`pip install benchopt`), and then run `benchopt run benchmarks/benchmark_ica --n-repetitions 5`.

## Orthogonal CNN experiment

You can run the experiment with orthogonal convolutions with `python 2_cifar10_resnet.py` or `python 2_cifar10_vgg16.py` in the folder `experiments/`, which will produce output files in the folder `experiments/outputs`.

A single instance of the orthogonal convolution experiment, can be run with `python cifar10_experiment.py`, see `__name__ == '__main__'` therein.


## Figures

To reproduce the plots using the provided convergence data, you can use the `makefile` in the folder `figures/`.
