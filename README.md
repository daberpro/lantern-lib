# Lantern
### C++ Lib for deep learning
![Test Build](https://github.com/daberpro/lantern-lib/actions/workflows/cmake-single-platform.yml/badge.svg)
![GitHub release](https://img.shields.io/github/v/release/daberpro/lantern-lib?include_prereleases)
![GitHub](https://img.shields.io/github/license/daberpro/lantern-lib)
![Code Size](https://img.shields.io/github/languages/code-size/daberpro/lantern-lib)
![Language](https://img.shields.io/github/languages/top/daberpro/lantern-lib)
![GitHub issues](https://img.shields.io/github/issues/daberpro/lantern-lib)
![GitHub pull requests](https://img.shields.io/github/issues-pr/daberpro/lantern-lib)
![GitHub Repo stars](https://img.shields.io/github/stars/daberpro/lantern-lib)
![GitHub forks](https://img.shields.io/github/forks/daberpro/lantern-lib)

## About Lantern
The lantern library is a library for developing deep learning written using the c++ programming language, built on the arrayfire library as a library for processing tensors.

> **⚠️ Danger:** This is a critical warning message! \
> lantern-lib still in progress, and this lib still poor feature

## Getting Started
### Feed Forward Neural Network (FFN)
lantern has a classic neural network which establish using perceptron neural network \
and lantern has several type of optimization such as
- Gradient Descent (GD)
- Root Mean Squared Propagation (RMSProp)
- Adaptive Gradient Descent (AdaGrad)
- Adaptive Gradient Estimation (Adam)

lantern use Normal Distribution to initalize weight and bias \
lantern has two type optimization, the first optimization \
was using OPTIMIZE_VERSION and the second is MATRIX_OPTIMIZE \
to use it just define macro with OPTIMIZE_VERSION or MATRIX_OPTIMIZE \

the main different between OPTIMIZE_VERSION and MATRIZ_OPTIMIZE \
is the first use lantern utility vector to store and update the weights and bias \
for Feed Forward Neural Network, and the second use arrayfire as a tensor and update \
weights and bias via arrayfire