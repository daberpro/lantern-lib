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
	model.SetBatchSize(2);
	model.SetMaxTreshold(1e-08);

	model.AddInputLayer<Activation::NOTHING>(2);
	model.AddHiddenLayer<Activation::SWISH>(3);
	model.AddHiddenLayer<Activation::SWISH>(4);
	model.AddOutputLayer<Activation::SIGMOID>(1);

	model.InitModel();
	
	model.Train<double,af::array>(
		input,
		target,
		lantern::perceptron::loss::SumSquaredResidual,
		lantern::perceptron::loss::DerivativeSumSquaredResidual
	);
	model.ShowParameters();
	
	
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

