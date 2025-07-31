#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"


int main(){

	af::info();
	std::cout << "\n\n";
	af::setSeed(static_cast<uint64_t>(std::time(nullptr)));

	lantern::ffn::layer::Layer layer;
	layer.Add<lantern::ffn::node::NodeType::NOTHING>(2); // layer 0 (Input)
	layer.Add<lantern::ffn::node::NodeType::SWISH>(6); // layer 1 (Hidden 1)
	layer.Add<lantern::ffn::node::NodeType::SWISH>(4); // layer 2 (Hidden 2)
	layer.Add<lantern::ffn::node::NodeType::SIGMOID>(1); // layer 3 (Output)

	lantern::utility::Vector<af::array> outputs;
	lantern::utility::Vector<af::array> parameters;
	lantern::utility::Vector<af::array> prev_gradient;
	lantern::ffn::optimizer::AdaptiveMomentEstimation optimizer;

	double input_data[] = {1,1,0,0,1,0,1,0};
	double target_data[] = {0,1,1,0};

	af::array input = af::array(4,2,input_data);
	af::array target = af::array(4,1,target_data);

	lantern::ffn::feedforward::Initialize(
		layer,
		parameters,
		prev_gradient,
		outputs,
		optimizer
	);
	
	prev_gradient.push_back(af::array());

	double loss = 0;
	uint32_t index;
	af::array selected_target;
	for(uint32_t iter = 1; iter < 10000; iter++){

		index = iter % 4;
		outputs[0] = input.row(index).T();
		selected_target = target.row(index).T();

		lantern::ffn::feedforward::FeedForward(
			layer,
			outputs,
			parameters
		);
		
		loss = lantern::loss::SumSquareResidual(outputs.back(),selected_target);
		std::cout << "Loss : " << loss << '\n';

		if(loss <= 1e-07){
			break;
		}

		prev_gradient.back() = lantern::derivative::SumSquareResidual(outputs.back(),selected_target);

		lantern::ffn::backprop::Backpropagate(
			layer,
			parameters,
			prev_gradient,
			outputs,
			optimizer
		);

	}

	std::cout << parameters << '\n';

	for(uint32_t i = 0; i < 4; i++){
		outputs[0] = input.row(i).T();
		lantern::ffn::feedforward::FeedForward(
			layer,
			outputs,
			parameters
		);
		std::cout << "\nPrediction: \n " << outputs.back();
		std::cout << "Target: \n " << target.row(i);
	}

	return 0;
}