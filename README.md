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
### Build From Scratch
lantern was build on it's own component, but for several feature like plot and tensor
lantern use external dependecies like arrayfire and matplot++
to build lantern yout must have
1. ArrayFire
2. Matplot++

lantern use CMake for build, so make sure you have cmake already installed on your 
device, the cmake version was use on this library is 3.31, and this lib was develop on Windows OS

### Feed Forward Neural Network (FFN)
lantern has a classic neural network which establish using perceptron neural network \
and lantern has several type of optimization such as
- Gradient Descent (GD)
- Root Mean Squared Propagation (RMSProp)
- Adaptive Gradient Descent (AdaGrad)
- Adaptive Gradient Estimation (Adam)

lantern use Normal Distribution to initalize weight and bias
lantern has two type optimization, the first optimization
was using OPTIMIZE_VERSION and the second is MATRIX_OPTIMIZE
to use it just define macro with OPTIMIZE_VERSION or MATRIX_OPTIMIZE

the main different between OPTIMIZE_VERSION and MATRIZ_OPTIMIZE
is the first use lantern utility vector to store and update the weights and bias
for Feed Forward Neural Network, and the second use arrayfire as a tensor and update
weights and bias via arrayfire

#### Simple Example
this is the easiest way to use lantern-lib with wrapper FeedForwardNetwork class

Example of using MATRIX_OPTIMIZE
```cpp
#include "pch.h"
#define MATRIX_OPTIMIZE
#include <Vector.h>
#include <Logging.h>
#include "FeedForwardNetwork/FeedForwardNetwork.h"

#define Activation lantern::perceptron::Activation
#define Optimizer lantern::perceptron::optimizer

int main() {

	double input_data[8] = {1,1,0,0,1,0,1,0};
	double target_data[4] = {0,1,1,0};
	af::array input = af::array(4,2,input_data);
	af::array target = af::array(4,1,target_data);

	lantern::FeedForwardNetwork<Optimizer::AdaptiveMomentEstimation> model;
	model.SetEpoch(10000);
	model.SetBatchSize(3);
	model.SetMaxTreshold(1e-06);

	model.AddInputLayer<Activation::NOTHING>(2);
	model.AddHiddenLayer<Activation::SWISH>(3);
	model.AddOutputLayer<Activation::SIGMOID>(1);

	model.InitModel();
	// model.ShowParameters();

	model.Train(
		input,
		target
	);

	lantern::utility::Vector<af::array> predict_result;
	model.Predict(
		input,
		predict_result
	);
	
	std::cout << std::string(70,'=') << '\n';
	for(uint32_t i = 0; i < input.dims(0); i++){
		std::cout << af::toString("Input : ",input.row(i),16,true) << '\n';
		std::cout << af::toString("Output : ",predict_result[i],16,true) << '\n';
		std::cout << std::string(70,'=') << '\n';
	}
    
    return 0;
}
```

#### Advance Example
this is an advance example for you which want to create perceptron model
from scratch using lantern-lib

Example of using MATRIX_OPTIMIZE
```cpp
#include "../../pch.h"
#define MATRIX_OPTIMIZE
#include "../../Headers/Vector.h"
#include "../../FeedForwardNetwork/FeedForwardNetwork.h"

int main() {

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();	
	std::atomic<int32_t> iteration(0);
	std::atomic<double> loss(0);
	std::atomic<bool> stop(false);

	std::jthread learning([&]() -> void {
		
		double input[6][2] = {
			{0.925925926,	0.148148148 }, // {170.0, 65.0} Male
			{0.851851852,	0.037037037 }, // {160.0, 50.0} Female
			{0.962962963,	0.185185185 }, // {175.0, 70.0} Male
			{0.814814815,	0.000000000 }, // {155.0, 45.0} Female
			{1.000000000,	0.222222222 }, // {180.0, 75.0} Male
			{0.888888889,	0.074074074 }  // {165.0, 55.0} Female
		};
		double target[6] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
		
		lantern::perceptron::Perceptron i1(input[0][0], "Height");
		lantern::perceptron::Perceptron i2(input[0][1], "Weight");
		lantern::perceptron::Perceptron h11("h11");
		lantern::perceptron::Perceptron h12("h12");
		lantern::perceptron::Perceptron h13("h13");
		lantern::perceptron::Perceptron h21("h21");
		lantern::perceptron::Perceptron h22("h22");
		lantern::perceptron::Perceptron h23("h23");
		lantern::perceptron::Perceptron o1("Output");
	
		lantern::perceptron::activation::Swish(h11, i1, i2);
		lantern::perceptron::activation::Swish(h12, i1, i2);
		lantern::perceptron::activation::Swish(h13, i1, i2);
		lantern::perceptron::activation::Swish(h21, h11, h12, h13);
		lantern::perceptron::activation::Swish(h22, h11, h12, h13);
		lantern::perceptron::activation::Swish(h23, h11, h12, h13);
		lantern::perceptron::activation::Sigmoid(o1, h21, h22, h23);
	
		lantern::perceptron::Layer layer;
		layer.SetLayer<3>(o1);
		layer.SetLayer<2>(h21, h22, h23);
		layer.SetLayer<1>(h11, h12, h13);
		layer.SetLayer<0>(i1, i2);
	
		lantern::utility::Vector<af::array> parameters;
		lantern::utility::Vector<af::array> gradient_based_parameters;
		lantern::utility::Vector<af::array> outputs;
		lantern::utility::Vector<lantern::perceptron::Activation> operators;
		lantern::utility::Vector<af::array> batch_gradient;
	
		lantern::perceptron::optimizer::AdaptiveMomentEstimation adam;
		lantern::perceptron::FeedForward(layer, parameters, gradient_based_parameters, operators, outputs, adam, batch_gradient);
		
		gradient_based_parameters.push_back(af::constant(1.0f, 1, f64));

		std::random_device rd;
		std::mt19937 rg(rd());
		std::uniform_int_distribution<> dis(0, 5);
	
		uint32_t i = 0, batch_iter = 0, batch_size = 3;
		af::array output;
		
		while (iteration < 10000) {
			i = dis(rg);
			parameters[0] = af::array(2, 1, input[i]);
			output = af::array(1, 1, &target[i]);
	
			lantern::perceptron::FeedForward(parameters, operators, outputs);
			loss = lantern::perceptron::loss::SumSquaredResidual(outputs.back(), output);
			
			if(batch_size == 1)
			{
				gradient_based_parameters.back() = lantern::perceptron::loss::DerivativeSumSquaredResidual(outputs.back(), output);
				lantern::perceptron::BackPropagation(parameters, gradient_based_parameters, operators, outputs, adam);
			}
			else
			{
				if(batch_iter % batch_size == 0 && batch_iter != 0){
					
					/**
					 * calculate average gradient 
					 * and reset batch_iter to 0
					 * then update each parameter
					 * using optimizer 
					 */
					for(int32_t p = parameters.size() - 1; p > 0; p--){
						batch_gradient[p] /= batch_size;
						batch_gradient[p].eval();
						parameters[p] -= adam.GetDelta(batch_gradient[p],p);
						parameters[p].eval();
					}
					batch_iter = 0;

				}else{
					gradient_based_parameters.back() = lantern::perceptron::loss::DerivativeSumSquaredResidual(outputs.back(), output);
					lantern::perceptron::CalculateGradient(
						parameters, 
						gradient_based_parameters, 
						operators, 
						outputs, 
						adam,
						batch_gradient
					);
				}
			}

			if(loss <= 0.0001){
				break;
			}
	
			iteration++;
			batch_iter++;
		}

		stop = true;

	});

	std::jthread progress([&]()-> void {
		while(!stop){
			lantern::perceptron::ProgressBar(iteration,10000);
			std::cout << std::fixed << std::setw(5) << " | Loss : " << std::setw(16) << std::setprecision(16) << loss << ", Iteration : " << std::setw(5) << iteration;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			std::cout << std::flush;
		}

		if(stop && iteration < 10000){
			lantern::perceptron::ProgressBar(10000,10000);
			std::cout << std::fixed << std::setw(5) << " | Loss : " << std::setw(16) << std::setprecision(16) << loss << ", Iteration : " << std::setw(5) << iteration;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			std::cout << std::flush;
		}
	});


    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_running_times = end - start;
    std::cout << "\n";
	std::cout << "Total time: " << total_running_times.count() << "s\n";
    
    return 0;
}


```
