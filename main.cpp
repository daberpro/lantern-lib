#define MATRIX_OPTIMIZE
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

	// af::array inputs(4,1);
	// af::array outputs(4,1);

	// lantern::FFN<lantern::MeanSquaredError> model;
	// model.Add<lantern::perceptron::input::Scalar>(2);
	// model.Add<lantern::perceptron::activation::Sigmoid>(4);
	// model.Add<lantern::perceptron::activation::Sigmoid>(1);

	// model.Train(
	// 	inputs,
	// 	outputs
	// );

	// model.Test(
	// 	inputs
	// );

	// double inp = 1.0f;
	// lantern::perceptron::Perceptron i1(&inp,"i1");
	// lantern::perceptron::Perceptron h1("h1");
	// lantern::perceptron::Perceptron h2("h2");
	// lantern::perceptron::Perceptron h3("h3");
	// lantern::perceptron::Perceptron o1("o1");

	// lantern::perceptron::activation::Sigmoid(h1,i1);
	// lantern::perceptron::activation::Sigmoid(h2,h1);
	// lantern::perceptron::activation::Sigmoid(h3,h2);
	// lantern::perceptron::activation::Sigmoid(o1,h3);

	// lantern::perceptron::SetLayer<1>(h1);
	// lantern::perceptron::SetLayer<2>(h2);
	// lantern::perceptron::SetLayer<3>(h3);
	// lantern::perceptron::SetLayer<4>(o1);

	// lantern::utility::Vector<af::array> parameters;
	// lantern::utility::Vector<af::array> gradient_based_parameters;
	// lantern::utility::Vector<af::array> outputs;
	// lantern::utility::Vector<lantern::perceptron::Activation> operators;

	// lantern::perceptron::FeedForward(
	// 	&o1,
	// 	parameters,
	// 	gradient_based_parameters, 
	// 	operators,
	// 	outputs
	// );

	// std::cout << "Parameters : \n";
	// std::cout << std::string(50,'=') << "\n";
	// for(auto& v : parameters){
	// 	std::cout << v << "\n";
	// }
	// std::cout << std::string(50,'=') << "\n";

	// std::cout << "Outputs : \n";
	// std::cout << std::string(50,'=') << "\n";
	// for(auto& v : outputs){
	// 	std::cout << v << "\n";
	// }
	// std::cout << std::string(50,'=') << "\n";

	// std::cout << "Gradient Based : \n";
	// std::cout << std::string(50,'=') << "\n";
	// for(auto& v : gradient_based_parameters){
	// 	std::cout << v << "\n";
	// }
	// std::cout << std::string(50,'=') << "\n";

	// lantern::perceptron::optimizer::GradientDescent gd(1.0f);

	// gradient_based_parameters.push_back(af::constant(1.0f,1,f64));
	// lantern::perceptron::BackPropagation(
	// 	parameters,
	// 	gradient_based_parameters, 
	// 	operators,
	// 	outputs,
	// 	gd
	// );

	// std::cout << "\n\nParameters : \n";
	// std::cout << std::string(50,'=') << "\n";
	// for(auto& v : parameters){
	// 	std::cout << v << "\n";
	// }
	// std::cout << std::string(50,'=') << "\n";

	// std::cout << "Outputs : \n";
	// std::cout << std::string(50,'=') << "\n";
	// for(auto& v : outputs){
	// 	std::cout << v << "\n";
	// }
	// std::cout << std::string(50,'=') << "\n";

	// std::cout << "Gradient Based : \n";
	// std::cout << std::string(50,'=') << "\n";
	// for(auto& v : gradient_based_parameters){
	// 	std::cout << v << "\n";
	// }
	// std::cout << std::string(50,'=') << "\n";

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

	lantern::perceptron::FeedForward(
		&o1,
		parameters,
		gradient_based_parameters, 
		operators,
		outputs
	);

	gradient_based_parameters.push_back(af::constant(1.0f,1,f64));
	lantern::perceptron::optimizer::GradientDescent gd(0.01);

	std::random_device rd;
	std::mt19937 rg(rd());
	std::uniform_int_distribution<> dis(0,3);

	uint32_t i = 0, iter = 0;
	while(true){

		i = dis(rg);
		parameters[0](0) = input[i][0];
		parameters[0](1) = input[i][1];

		lantern::perceptron::FeedForward(
			parameters,
			operators,
			outputs
		);
	
		loss = pow(target[i] - outputs.back()(0).scalar<double>(),2);
		std::cout << "Input : " << parameters[0](0).scalar<double>() << "," << parameters[0](1).scalar<double>() << " | Predict: " << outputs.back()(0).scalar<double>() << ", Target: " << target[i] << " | Loss " << loss << "\n";
		gradient_based_parameters.back()(0) = -2 * (target[i] - outputs.back()(0).scalar<double>()); 

		lantern::perceptron::BackPropagation(
			parameters,
			gradient_based_parameters, 
			operators,
			outputs,
			gd
		);

		if(loss <= 0.001 && iter % 100 == 0){
			break;
		}
		iter++;
	}

	
	std::cout << "\n\nParameters : \n";
	std::cout << std::string(50,'=') << "\n";
	for(auto& v : parameters){
		std::cout << v << "\n";
	}
	std::cout << std::string(50,'=') << "\n";

	std::cout << "Outputs : \n";
	std::cout << std::string(50,'=') << "\n";
	for(auto& v : outputs){
		std::cout << v << "\n";
	}
	std::cout << std::string(50,'=') << "\n";

	std::cout << "Gradient Based : \n";
	std::cout << std::string(50,'=') << "\n";
	for(auto& v : gradient_based_parameters){
		std::cout << v << "\n";
	}
	std::cout << std::string(50,'=') << "\n";
	
	// std::chrono::time_point start = std::chrono::high_resolution_clock::now();

	// uint32_t i = 0, iter = 0;
	// // lantern::perceptron::optimizer::StochasticGradientDescentWithMomentum sgdm(0.01f, 0.99f);
	// // lantern::perceptron::optimizer::AdaptiveGradientDescent adagrad(0.01f);
	// // lantern::perceptron::optimizer::RootMeanSquarePropagation rmsprop(0.01f, 0.99f, 1e-8);
	// lantern::perceptron::optimizer::AdaptiveMomentEstimation adam(0.01f, 0.9f, 0.999f, 1e-8);
	// while(true){
		
	// 	i = dis(rg);
	// 	i1.value = input_2[i][0];
	// 	i2.value = input_2[i][1];
		
	// 	lantern::perceptron::PerceptronFeedForward(fix_position_node);
		
	// 	loss = pow(target_2[i] - o1.value,2);
	// 	std::cout << "Predict: " << o1.value << ", Target: " << target_2[i] << " | Loss " << loss << "\n";
	// 	o1.gradient_based_input[0] = -2 * (target_2[i] - o1.value);
		
	// 	lantern::perceptron::BackPropagation(o1,adam);
		
	// 	if(loss <= 0.001 && iter % 100 == 0){
	// 		break;
	// 	}
	// 	iter++;
	// }
	
	// std::cout << "\nResult Testing \n";
	// for(uint32_t j = 0; j < 4; j++){
	// 	i1.value = input_2[j][0];
	// 	i2.value = input_2[j][1];
	// 	lantern::perceptron::PerceptronFeedForward(fix_position_node);
	// 	std::cout << "input [i1: " << i1.value << ", i2: " << i2.value << "], predict : " << o1.value << ", target : " << target_2[j] << "\n"; 
	// }
	
	// lantern::print(o1,h4,h3,h2,h1,i2,i1);
	
	// std::chrono::time_point end = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double> time_complete = (end - start);
	// std::chrono::milliseconds in_mills = std::chrono::duration_cast<std::chrono::milliseconds>(time_complete); 
	// std::cout << std::string(50,'=') << "\n";
	// std::cout << "Time complete on second " << time_complete.count() << "\n"; 
	// std::cout << "Time complete on milisecond " << in_mills.count() << "\n"; 
	// std::cout << std::string(50,'=') << "\n";

	return EXIT_SUCCESS;
}