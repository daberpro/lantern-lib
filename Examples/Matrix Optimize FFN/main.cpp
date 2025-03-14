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
	
	lantern::utility::Vector<af::array> parameters;
	lantern::utility::Vector<af::array> gradient_based_parameters;
	lantern::utility::Vector<af::array> outputs;
	lantern::utility::Vector<lantern::perceptron::Activation> operators;

    lantern::perceptron::Layer layer;
    layer.SetLayer<2>(o1);
    layer.SetLayer<1>(h1,h2,h3,h4);
    layer.SetLayer<0>(i1,i2);

	lantern::perceptron::optimizer::AdaptiveMomentEstimation adam;

	lantern::perceptron::FeedForward(
		layer,
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
	while(iter < 10000){

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
		
		// if(loss <= 0.001 && iter % 100 == 0){
		// 	break;
		// }
		iter++;
	}

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_running_times = end - start;
    std::cout << std::string(70,'=') << '\n';
    std::cout << "Total time : " << total_running_times.count() << "s\n";
    std::cout << std::string(70,'=') << '\n';
	
    return 0;
}
