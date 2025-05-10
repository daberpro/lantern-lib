#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"


int main(){

	af::info();
	std::cout << "\n\n";
	af::setSeed(static_cast<uint64_t>(std::time(nullptr)));

	double input_data[] = {
		0.925925926,	0.148148148, // {170.0, 65.0} Male
		0.851851852,	0.037037037, // {160.0, 50.0} Female
		0.962962963,	0.185185185, // {175.0, 70.0} Male
		0.814814815,	0.000000000, // {155.0, 45.0} Female
		1.000000000,	0.222222222, // {180.0, 75.0} Male
		0.888888889,	0.074074074  // {165.0, 55.0} Female
	};
	double target_data[] = {
		1.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		0.0, 1.0
	};

	af::array input = af::array(2, 6, input_data);
	af::array target = af::array(2, 6, target_data);

	input = input.T();
	target = target.T();

	lantern::layer::Layer layer;
	layer.Add<lantern::node::NodeType::NOTHING>(2);
	layer.Add<lantern::node::NodeType::SWISH>(6);
	layer.Add<lantern::node::NodeType::SWISH>(6);
	layer.Add<lantern::node::NodeType::LINEAR>(2);

	lantern::optimizer::AdaptiveMomentEstimation optimizer;

	lantern::feedforward::FeedForwardNetwork model;
	model.SetInput(&input);
	model.SetTarget(&target);
	model.SetLayer(&layer);
	model.SetEachClassSize({3,2});
	model.SetMinimumTreshold(1e-08);
	model.SetEpoch(100);
	model.Train<
		6,
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

	std::cout << test_results << '\n';

	return 0;
}