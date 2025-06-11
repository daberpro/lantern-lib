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

If you have any suggestions or improvements, feel free to contact me through the channels below.
| 📧 Email                   | 🗨️ Discord      |
|---------------------------|-----------------|
| daber.coding@gmail.com    | daberdev        |

## # Getting Started
### > Build From Scratch
lantern was build on it's own component, but for several feature like plot and tensor
lantern use external dependecies like arrayfire and matplot++
to build lantern yout must have
1. ArrayFire
2. Matplot++
3. HDF5 (Build with c++ enable)

lantern use CMake for build, so make sure you have cmake already installed on your 
device, the cmake version was use on this library is 4.0, and this lib was develop on Windows OS
using Visual Studio 2022

### > Feed Forward Neural Network (FFN)
lantern has a classic neural network which establish using perceptron neural network \
and lantern has several type of optimization such as
- Gradient Descent (GD)
- Root Mean Squared Propagation (RMSProp)
- Adaptive Gradient Descent (AdaGrad)
- Adaptive Gradient Estimation (Adam)

lantern use Normal Distribution to initalize weight and bias and optimize the initalize weights and bias using Xavier/Glorot Initalization

#### Example
this is an example of Multiple Class model, i know i should use the loss function BinaryCrossEntropy but using Cross Entropy also works well 

Example of Multiple Class using lantern-lib
```cpp
#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"

int main(){

	af::info();
	std::cout << "\n\n";
	af::setSeed(static_cast<uint64_t>(std::time(nullptr)));

	// 45 samples × 2 features = 90 elements
	double input_data[] = {
		// Class 0 Col 1
		1.0, 1.2, 0.8, 1.1, 1.3, 0.9, 1.2, 1.0, 0.7, 1.1, 1.0, 0.8, 1.2, 1.1, 0.9,
		// Class 1 Col 1
		3.0, 3.2, 2.9, 3.1, 3.3, 3.0, 3.1, 3.2, 3.0, 3.1, 2.9, 3.3, 3.0, 3.1, 2.8, 
		// Class 2 Col 1
		5.0, 5.2, 4.9, 5.1, 5.3, 5.0, 5.1, 4.8, 5.2, 5.0, 5.1, 4.9, 5.2, 5.0, 5.1, 
		
		// Class 0 Col 2
		2.0, 1.9, 2.2, 2.1, 2.3, 1.8, 2.4, 1.7, 2.0, 1.9, 2.1, 1.9, 2.2, 2.0, 2.1,
		// Class 1 Col 2
		3.5, 3.7, 3.4, 3.6, 3.8, 3.2, 3.9, 3.5, 3.3, 3.4, 3.6, 3.4, 3.7, 3.3, 3.5,
		// Class 2 Col 2
		1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 1.0, 1.1, 0.9, 0.8, 1.1, 1.2, 1.0, 1.2, 0.9
	};

	// 45 samples × 3 classes = 135 elements
	double target_data[] = {

		// Class 0: one-hot [1, 0, 0] Col 1
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		// Class 1: one-hot [0, 1, 0] Col 1
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		// Class 2: one-hot [0, 0, 1] Col 1
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		
		// Class 0: one-hot [1, 0, 0] Col 2
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		// Class 1: one-hot [0, 1, 0] Col 2
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
		// Class 2: one-hot [0, 0, 1] Col 2
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		
		// Class 0: one-hot [1, 0, 0] Col 3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		// Class 1: one-hot [0, 1, 0] Col 3
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		// Class 2: one-hot [0, 0, 1] Col 3
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
	};

	af::array input = af::array(45, 2, input_data);
	af::array target = af::array(45, 3, target_data);

	lantern::layer::Layer layer;
	layer.Add<lantern::node::NodeType::NOTHING>(2);
	layer.Add<lantern::node::NodeType::SWISH>(15);
	layer.Add<lantern::node::NodeType::SWISH>(15);
	layer.Add<lantern::node::NodeType::LINEAR>(3);

	lantern::optimizer::AdaptiveMomentEstimation optimizer;

	lantern::feedforward::FeedForwardNetwork model;
	model.SetInput(&input);
	model.SetTarget(&target);
	model.SetLayer(&layer);
	model.SetEachClassSize({15,15,14});
	model.SetMinimumTreshold(1e-08);
	model.Train<
		15,
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

	return 0;
}
```
