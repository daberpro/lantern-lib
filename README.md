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

## # Getting Started
### > Build From Scratch
lantern was build on it's own component, but for several feature like plot and tensor
lantern use external dependecies like arrayfire and matplot++
to build lantern yout must have
1. ArrayFire
2. Matplot++

lantern use CMake for build, so make sure you have cmake already installed on your 
device, the cmake version was use on this library is 3.31, and this lib was develop on Windows OS

### > Feed Forward Neural Network (FFN)
lantern has a classic neural network which establish using perceptron neural network \
and lantern has several type of optimization such as
- Gradient Descent (GD)
- Root Mean Squared Propagation (RMSProp)
- Adaptive Gradient Descent (AdaGrad)
- Adaptive Gradient Estimation (Adam)

## # Example Code
this a simple example of using lantern-lib to classify gender by weight and height of man and woman
```cpp
#include "pch.h"
#include "Headers/Logging.h"
#include "FeedForwardNetwork/FeedForwardNetwork.h"

int main(){

	af::info();
	std::cout << "\n\n";
	af::setSeed(static_cast<uint64_t>(std::time(nullptr)));

	double input_data[] = {
		0.925925926,	0.148148148, // {170.0, 65.0} Male
		0.851851852,	0.037037037, // {160.0, 50.0} Female
		0.962962963,	0.185185185, // {175.0, 70.0} Male
		0.814814815,	0.000000000, // {155.0, 45.0} Female
		1.000000000,	0.222222222, // {180.0, 75.0} Male
		0.888888889,	0.074074074  // {165.0, 55.0} Female
	};
	double target_data[] = {
		1.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 1.0
	};

	af::array input = af::array(2, 6, input_data);
	af::array target = af::array(2, 6, target_data);

	input = input.T();
	target = target.T();

	lantern::layer::Layer layer;
	layer.Add<lantern::node::NodeType::NOTHING>(2);
	layer.Add<lantern::node::NodeType::SWISH>(6);
	layer.Add<lantern::node::NodeType::SWISH>(6);
	layer.Add<lantern::node::NodeType::LINEAR>(2);

	lantern::optimizer::AdaptiveMomentEstimation optimizer;

	lantern::feedforward::FeedForwardNetwork model;
	model.SetInput(&input);
	model.SetTarget(&target);
	model.SetLayer(&layer);
	model.SetEachClassSize({3,2});
	model.SetMinimumTreshold(1e-08);
	model.SetEpoch(100);
	model.Train<
		6,
		lantern::optimizer::AdaptiveMomentEstimation,
		double,
		af::array,
		af::array
	>(
		optimizer,
		lantern::loss::CrossEntropy,
		lantern::derivative::CrossEntropySoftMax,
		lantern::probability::SoftMax
	);

	af::array test_results;
	model.Predict<af::array>(
		input,
		test_results,
		lantern::probability::SoftMax
	);

	for(auto& ar : model.GetParameters()){
		std::cout << ar << '\n';
	}
	std::cout << test_results << '\n';

	return 0;
}
```
