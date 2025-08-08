#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"


int main(){

	af::info();
	std::cout << "\n\n";
	af::setSeed(static_cast<uint64_t>(std::time(nullptr)));

	// 45 samples 2 features = 90 elements
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

	// 45 samples 3 classes = 135 elements
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

	/**
	 * Create Layer with 
	 * 2 -> 15 -> 15 -> 3
	 * NOTHING -> SWISH -> SWISH -> LINEAR
	 */
	lantern::ffn::layer::Layer layer;
	layer.Add<lantern::ffn::node::NodeType::NOTHING>(2);
	layer.Add<lantern::ffn::node::NodeType::SWISH>(15);
	layer.Add<lantern::ffn::node::NodeType::SWISH>(15);
	layer.Add<lantern::ffn::node::NodeType::LINEAR>(3);

	layer.PrintLayerInfo();

	/**
	 * Using lantern AdaptiveMomenEstimation (ADAM) optimizer
	 * with default value
	 */
	lantern::ffn::optimizer::AdaptiveMomentEstimation optimizer;

	/**
	 * Create model using FeedForwardNetwork Class and pass
	 * - input pointer 
	 * - target pointer
	 * - layer pointer
	 * - size of each class for example (15 class 1, 15 class 2, 15 class 3)
	 * 	 because the class index will start from 0 we need to adjust the last class size
	 * 	 to substract it with 1
	 * - minum treshold when the loss in training are <= this value, the training will stop
	 * - number of epoch
	 */
	lantern::feedforward::FeedForwardNetwork model(
		&input,
		&target,
		&layer,
		{15,15,15-1},
		1e-08,
		200
	);

	/**
	 * ==========================================================================
	 * Train the model
	 * ==========================================================================
	 */

	/**
	 * Train the model, the template value 15 is a number of batch will use in training
	 * and then the pass value is
	 * - optimizer that we have already created
	 * - loss function, we use CrossEntropy because we want to do classify
	 * - output function (by default is Linear) we want the output of the model was feed to SoftMax First
	 * 	 this happend because our model just use Linear at the output layer, because if we just use SoftMax directly
	 * 	 the model cannot convergen
	 */
	model.Train<15>(
		optimizer,
		lantern::loss::CrossEntropy,
		lantern::derivative::CrossEntropySoftMax,
		lantern::probability::SoftMax
	);

	/**
	 * Create test result to show the result
	 * we just feed the model with sam input to see all output
	 * the Predict() function just take input data, and an af::array to save the output result prediction
	 * and the output function (default is Linear) but we use SoftMax
	 */
	af::array test_results;
	model.Predict(
		input,
		test_results,
		lantern::probability::SoftMax
	);
	std::cout << test_results << '\n'; // then we print the result, which the overload function for this define in Logging.h
	
	/**
	 * After that we save the model, lantern by default use HDF5 to manage
	 * all data from model, teh value we pass to the SaveModel() function is
	 * - path where the model will save to with extension .h5
	 * - then the name of output function we use, u can use LANTERN_GET_FUNC_NAME(lantern::probability::SoftMax) macro
	 */
	model.SaveModel(
		"Result.h5",
		"lantern::probability::SoftMax"
	);

	/**
	 * ==========================================================================
	 * Load the model
	 * ==========================================================================
	 */

	/**
	 * Load the model we have already save
	 */
	model.LoadModel("Result.h5");
	auto l = model.GetLayer();
	af::array predict_result;
	/**
	 * Then predict again 
	 */
	model.Predict(
		input,
		predict_result,
		lantern::probability::SoftMax
	);
	std::cout << predict_result << '\n';

	return 0;
}