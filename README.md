# Landing algorithm on the Stiefel manifold

This repository contains the code for the landing algorithm under the orthogonal Stiefel manifold constraint without the use of retractions. 

The algorithm is implemented as a PyTorch optimizer; see `landing-stiefel/landing_stiefel/optimizer.py`.

You can find the paper [here](https://arxiv.org/abs/2303.16510).

![Landing diagram](https://github.com/simonvary/landing-stiefel/blob/main/figures/landing_diagram.png?raw=true | width=40)

## ICA experiment

To run the ICA experiment, you need to install benchopt (`pip install benchopt`), and then run `benchopt run benchmarks/benchmark_ica --n-repetitions 5`.

## Online PCA and Orthogonal CNN experiments

To run the experiment with online PCA and the with the training of orthogonal CNNs, run  `python 1_pca.py` and `python 2_cifar10_resnet.py` in the folder `experiments/`. This will generate output files in the folder `experiments/outputs`.

A single instance of the experiments can be run with `python pca_experiment.py` and `python cifar10_experiment.py`. To run a single instance of the online PCA you can also see the notebook `example_pca.ipynb`.

## Figures

To reproduce the plots using the provided convergence data, you can use the `makefile` in the folder `figures/`.

## Cite

If you use this code please cite:
```
@misc{Ablin2023Infeasible,
  title = {{Infeasible Deterministic, Stochastic, and Variance-Reduction Algorithms for Optimization under Orthogonality Constraints}},
  author = {Ablin, Pierre and Vary, Simon and Gao, Bin and Absil, P.-A.},
  year = {2023},
  month = mar,
  eprint = {arXiv:2303.16510}
}
```
