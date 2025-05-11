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

	lantern::utility::Vector<uint32_t> batch_index;
	lantern::data::GetRandomSampleClassIndex<15>(batch_index,15,15,15 - 1);

	lantern::layer::Layer layer;
	layer.Add<lantern::node::NodeType::NOTHING>(2);
	layer.Add<lantern::node::NodeType::SWISH>(15);
	layer.Add<lantern::node::NodeType::SWISH>(15);
	layer.Add<lantern::node::NodeType::LINEAR>(3);

	lantern::utility::Vector<af::array> parameters;
	lantern::utility::Vector<af::array> prev_gradient;
	lantern::utility::Vector<af::array> outputs;
<<<<<<< HEAD
	lantern::optimizer::AdaptiveMomentEstimation optimizer;
=======
	lantern::optimizer::AdaptiveGradientDescent optimizer;
>>>>>>> 992cc9276e65bff7d6fb8603d03962aa3937c710

	lantern::feedforward::Initialize(
		layer,
		parameters,
		prev_gradient,
		outputs,
		optimizer
	);

	uint32_t epoch = 1000;
	uint32_t current_iter = 0;
	uint32_t batch_size = 15;	
	double loss = 1;
	af::array output, target_output;
	
	prev_gradient.push_back(af::array());
	while(current_iter < epoch){

		for(auto& selected_index : batch_index){

			outputs[0] = input.row(selected_index).T();
			target_output = target.row(selected_index).T();
			lantern::feedforward::FeedForward(
				layer,
				outputs,
				parameters
			);

			output = lantern::probability::SoftMax(outputs.back());
			loss = lantern::loss::CrossEntropy(output, target_output) / batch_size;
			std::cout << "Loss : " << loss << '\n';

			prev_gradient.back() = lantern::derivative::CrossEntropySoftMax(output, target_output);
			lantern::backprop::Backpropagate(
				layer,
				parameters,
				prev_gradient,
				outputs,
				optimizer,
				batch_size
<<<<<<< HEAD
				// ,lantern::regularization::L1Regularization
				// ,0.7	
=======
>>>>>>> 992cc9276e65bff7d6fb8603d03962aa3937c710
			);
		
		}

		lantern::data::GetRandomSampleClassIndex<15>(batch_index,15,15,15 - 1);
		current_iter++;

	}

	for(uint32_t i = 0; i < input.dims(0); i++){
		outputs[0] = input.row(i).T();
		lantern::feedforward::FeedForward(
			layer,
			outputs,
			parameters
		);
		output = lantern::probability::SoftMax(outputs.back());
		std::cout << "\nPrediction: \n " << output;
		std::cout << "Target: \n " << target.row(i);
	}

	return 0;
}