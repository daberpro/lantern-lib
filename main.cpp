#include "pch.h"
// #include <ReverseMode.h>
// #include <FeedForward.h>
#include <Logging.h>
#include <Vector.h>
#include <arrayfire.h>
#include <FeedForwardNetwork.h>
#include <random>
#include <chrono>

int main(){	

	af::info();

	// af::array inputs(4,1);
	// af::array outputs(4,1);

	// latern::FFN<latern::MeanSquaredError> model;
	// model.Add<latern::perceptron::input::Scalar>(2);
	// model.Add<latern::perceptron::activation::Sigmoid>(4);
	// model.Add<latern::perceptron::activation::Sigmoid>(1);

	// model.Train(
	// 	inputs,
	// 	outputs
	// );

	// model.Test(
	// 	inputs
	// );

	double input_value[] = {
		1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f
	};

	double target_value[] = {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};
	af::array target(4,1,target_value);
	af::array input(4,2,input_value);

	input = input.T();

	latern::perceptron::Perceptron i1(1.0,"i1");
	latern::perceptron::Perceptron i2(1.0,"i2");
	latern::perceptron::Perceptron h1("h1");
	latern::perceptron::Perceptron h2("h2");
	latern::perceptron::Perceptron h3("h3");
	latern::perceptron::Perceptron h4("h4");
	latern::perceptron::Perceptron o1("o1");

	latern::perceptron::activation::Swish(h1,i1,i2);
	latern::perceptron::activation::Swish(h2,i1,i2);
	latern::perceptron::activation::Swish(h3,i1,i2);
	latern::perceptron::activation::Swish(h4,i1,i2);
	latern::perceptron::activation::Sigmoid(o1,h4,h3,h2,h1);

	latern::utility::Vector<latern::perceptron::Perceptron* > fix_position_node;
	latern::perceptron::PerceptronFeedForward(&o1,fix_position_node);

	// latern::print(o1,h3,h2,h1,i1);
	// latern::perceptron::BackPropagation(o1);
	// latern::print(o1,h3,h2,h1,i1);

	double input_2[4][2] = {
		{1.0f, 1.0f},
		{1.0f, 0.0f},
		{0.0f, 1.0f},
		{0.0f, 0.0f}
	}; 
	double target_2[4] = {0.0,1.0,1.0,0.0};
	double loss = 0; 

	std::random_device rd;
	std::mt19937 rg(rd());
	std::uniform_int_distribution<> dis(0,3);

	std::chrono::time_point start = std::chrono::high_resolution_clock::now();

	uint32_t i = 0, iter = 0;
	latern::perceptron::optimizer::StochasticGradientDescentWithMomentum sgdm(0.01f, 0.99f);
	// latern::perceptron::optimizer::AdaptiveGradientDescent adagrad(0.01f);
	// latern::perceptron::optimizer::RootMeanSquarePropagation rmsprop(0.01f, 0.99f, 1e-8);
	// latern::perceptron::optimizer::AdaptiveMomentEstimation adam(0.01f, 0.9f, 0.999f, 1e-8);
	while(true){
		
		i = dis(rg);
		i1.value = input_2[i][0];
		i2.value = input_2[i][1];
		
		latern::perceptron::PerceptronFeedForward(fix_position_node);
		
		loss = pow(target_2[i] - o1.value,2);
		std::cout << "Predict: " << o1.value << ", Target: " << target_2[i] << " | Loss " << loss << "\n";
		o1.gradient_based_input(0,0) = -2 * (target_2[i] - o1.value);
		
		latern::perceptron::BackPropagation(o1,sgdm);
		
		if(loss <= 0.001 && iter % 100 == 0){
			break;
		}
		iter++;
	}
	
	std::cout << "\nResult Testing \n";
	for(uint32_t j = 0; j < 4; j++){
		i1.value = input_2[j][0];
		i2.value = input_2[j][1];
		latern::perceptron::PerceptronFeedForward(fix_position_node);
		std::cout << "input [i1: " << i1.value << ", i2: " << i2.value << "], predict : " << o1.value << ", target : " << target_2[j] << "\n"; 
	}
	
	latern::print(o1,h4,h3,h2,h1,i2,i1);
	
	std::chrono::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_complete = (end - start);
	std::chrono::milliseconds in_mills = std::chrono::duration_cast<std::chrono::milliseconds>(time_complete); 
	std::cout << std::string(50,'=') << "\n";
	std::cout << "Time complete on milisecond " << in_mills.count() << "\n"; 
	std::cout << std::string(50,'=') << "\n";

	return EXIT_SUCCESS;
}