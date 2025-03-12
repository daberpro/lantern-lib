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
to use it just define macro with OPTIMIZE_VERSION or MATRIX_OPTIMIZE

the main different between OPTIMIZE_VERSION and MATRIZ_OPTIMIZE \
is the first use lantern utility vector to store and update the weights and bias \
for Feed Forward Neural Network, and the second use arrayfire as a tensor and update \
weights and bias via arrayfire

Example of code using OPTIMIZE_VERSION
```cpp
#include "../../pch.h"
#define OPTIMIZE_VERSION
#include "../../Headers/Vector.h"
#include "../../FeedForwardNetwork/FeedForwardNetwork.h"

int main(){

    lantern::perceptron::Perceptron i1(1.0f,"i1");
    lantern::perceptron::Perceptron i2(1.0f,"i2");
    lantern::perceptron::Perceptron h1("h1");
    lantern::perceptron::Perceptron h2("h2");
    lantern::perceptron::Perceptron h3("h3");
    lantern::perceptron::Perceptron h4("h4");
    lantern::perceptron::Perceptron o1("o1");

    lantern::perceptron::activation::Swish(h1,i1,i2);
    lantern::perceptron::activation::Swish(h2,i1,i2);
    lantern::perceptron::activation::Swish(h3,i1,i2);
    lantern::perceptron::activation::Swish(h4,i1,i2);
    lantern::perceptron::activation::Sigmoid(o1,h1,h2,h3,h4);

    double inputs[4][2] = {
        {1,1},
        {1,0},
        {0,1},
        {0,0}
    };

    double outputs[4] = {
        0,
        1,
        1,
        0
    };

    lantern::utility::Vector<lantern::perceptron::Perceptron*> fix_position_nodes;
    lantern::perceptron::FeedForward(
        &o1,
        fix_position_nodes
    );

    uint32_t iter = 0, selected_index = 0;
    double loss = 0;
    lantern::perceptron::optimizer::AdaptiveMomentEstimation adam;
    // lantern::perceptron::optimizer::GradientDescent gd;

    std::random_device rd;
    std::mt19937 rg(rd());
    std::uniform_int_distribution<> rand(0,3);

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    while(true){

        selected_index = rand(rg);
        i1.value = inputs[selected_index][0];
        i2.value = inputs[selected_index][1];
        lantern::perceptron::FeedForward(fix_position_nodes);  
        loss = pow(outputs[selected_index] - o1.value,2);

        std::cout << "Inputs: ["        << inputs[selected_index][0] 
                  << ","                << inputs[selected_index][1] 
                  << "] | Predicted : " << o1.value 
                  << ", Target : "      << outputs[selected_index] 
                  << " | Loss : "       << loss 
                  << "\n";

        o1.gradient_based_input[0] = -2 * (outputs[selected_index] - o1.value);
        
        lantern::perceptron::BackPropagation(
            o1,
            adam
        );

        if(loss <= 0.001 && iter % 100 == 0){
            break;
        }

        iter++;

    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_running_times = end - start;
    std::cout << std::string(70,'=') << '\n';
    std::cout << "Total time : " << total_running_times.count() << "s\n";
    std::cout << std::string(70,'=') << '\n';


    return 0;
}
```

Example of using MATRIX_OPTIMIZE

```cpp
#include "../../pch.h"
#define MATRIX_OPTIMIZE
#include "../../Headers/Vector.h"
#include "../../FeedForwardNetwork/FeedForwardNetwork.h"

int main(){

    double input[4][2] = {
		{1.0f, 1.0f},
		{1.0f, 0.0f},
		{0.0f, 1.0f},
		{0.0f, 0.0f}
	}; 
	double target[4] = {0.0,1.0,1.0,0.0};
	double loss = 0;
	
	lantern::perceptron::Perceptron i1(&input[0][0],"i1");
	lantern::perceptron::Perceptron i2(&input[0][1],"i2");
	lantern::perceptron::Perceptron h1("h1");
	lantern::perceptron::Perceptron h2("h2");
	lantern::perceptron::Perceptron h3("h3");
	lantern::perceptron::Perceptron h4("h4");
	lantern::perceptron::Perceptron o1("o1");

	lantern::perceptron::activation::Swish(h1,i1,i2);
	lantern::perceptron::activation::Swish(h2,i1,i2);
	lantern::perceptron::activation::Swish(h3,i1,i2);
	lantern::perceptron::activation::Swish(h4,i1,i2);
	lantern::perceptron::activation::Sigmoid(o1,h4,h3,h2,h1);

	lantern::perceptron::SetLayer<1>(h1,h2,h3,h4);
	lantern::perceptron::SetLayer<2>(o1);
	
	lantern::utility::Vector<af::array> parameters;
	lantern::utility::Vector<af::array> gradient_based_parameters;
	lantern::utility::Vector<af::array> outputs;
	lantern::utility::Vector<lantern::perceptron::Activation> operators;

	lantern::perceptron::optimizer::AdaptiveMomentEstimation adam;

	lantern::perceptron::FeedForward(
		&o1,
		parameters,
		gradient_based_parameters, 
		operators,
		outputs,
		adam
	);

	gradient_based_parameters.push_back(af::constant(1.0f,1,f64));

	std::random_device rd;
	std::mt19937 rg(rd());
	std::uniform_int_distribution<> dis(0,3);

	uint32_t i = 0, iter = 0;
	af::array output;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
	while(true){

		i = dis(rg);
		parameters[0] = af::array(2,1,input[i]);
		output = af::array(1,1,&target[i]);

		lantern::perceptron::FeedForward(
			parameters,
			operators,
			outputs
		);
	
		loss = lantern::perceptron::loss::SumSquaredResidual(outputs.back(),output);
		std::cout << "Input : [" << parameters[0](0).scalar<double>() << "," << parameters[0](1).scalar<double>() << "] | Predict: " << outputs.back()(0).scalar<double>() << ", Target: " << target[i] << " | Loss " << loss << "\n";
		gradient_based_parameters.back() = lantern::perceptron::loss::DerivativeSumSquaredResidual(outputs.back(),output);
		
		lantern::perceptron::BackPropagation(
			parameters,
			gradient_based_parameters, 
			operators,
			outputs,
			adam
		);
		
		if(loss <= 0.001 && iter % 100 == 0){
			break;
		}
		iter++;
	}

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_running_times = end - start;
    std::cout << std::string(70,'=') << '\n';
    std::cout << "Total time : " << total_running_times.count() << "s\n";
    std::cout << std::string(70,'=') << '\n';
	
    return 0;
}
```
