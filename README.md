# KWaveSwift

A Swift port of the [k-Wave](http://www.k-wave.org/) acoustic simulation toolbox, providing
time-domain simulation of acoustic wave propagation using the k-space pseudospectral method.

Supports 1D, 2D, and 3D simulations of linear and nonlinear propagation in heterogeneous media
with power-law absorption. It is a clean-room reimplementation of the
[MATLAB k-Wave toolbox](https://github.com/ucl-bug/k-wave) (v1.4.1), targeting Apple silicon and
built on [MLX-Swift](https://github.com/ml-explore/mlx-swift) for GPU-accelerated array compute.
