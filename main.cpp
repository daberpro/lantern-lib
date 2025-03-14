#include "pch.h"
#define MATRIX_OPTIMIZE
#include <Vector.h>
#include <Logging.h>
#include "FeedForwardNetwork/FeedForwardNetwork.h"

af::array mult(const af::array& lhs,const af::array& rhs){
	return lhs * rhs;
}

int main() {

	
    double input[6][2] = {
        // {170.0, 65.0}, // Male
        // {160.0, 50.0}, // Female
        // {175.0, 70.0}, // Male
        // {155.0, 45.0}, // Female
        // {180.0, 75.0}, // Male
        // {165.0, 55.0}  // Female
		{0.925925926,	0.148148148 }, // Male
		{0.851851852,	0.037037037 }, // Female
		{0.962962963,	0.185185185 }, // Male
		{0.814814815,	0.000000000 }, // Female
		{1.000000000,	0.222222222 }, // Male
		{0.888888889,	0.074074074 }  // Female
    };
    double target[6] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    double loss = 0;

    lantern::perceptron::Perceptron i1(&input[0][0], "Height");
    lantern::perceptron::Perceptron i2(&input[0][1], "Weight");
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

    lantern::perceptron::optimizer::AdaptiveMomentEstimation adam;
    lantern::perceptron::FeedForward(layer, parameters, gradient_based_parameters, operators, outputs, adam);
    
    gradient_based_parameters.push_back(af::constant(1.0f, 1, f64));

    std::random_device rd;
    std::mt19937 rg(rd());
    std::uniform_int_distribution<> dis(0, 5);

    uint32_t i = 0, iter = 0;
    af::array output;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    while (iter < 10000) {
        i = dis(rg);
        parameters[0] = af::array(2, 1, input[i]);
        output = af::array(1, 1, &target[i]);

        lantern::perceptron::FeedForward(parameters, operators, outputs);
        loss = lantern::perceptron::loss::SumSquaredResidual(outputs.back(), output);

        std::cout << "Height: " << parameters[0](0).scalar<double>() << ", Weight: " << parameters[0](1).scalar<double>()
                  << " | Predicted: " << outputs.back()(0).scalar<double>()
                  << " | Target: " << target[i] << " | Loss: " << loss << "\n";

        gradient_based_parameters.back() = lantern::perceptron::loss::DerivativeSumSquaredResidual(outputs.back(), output);
        lantern::perceptron::BackPropagation(parameters, gradient_based_parameters, operators, outputs, adam);

        // if (loss <= 0.001 && iter % 500 == 0) {
        //     break;
        // }
        iter++;
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_running_times = end - start;
    std::cout << std::string(70, '=') << '\n';
    std::cout << "Total time: " << total_running_times.count() << "s\n";
    std::cout << std::string(70, '=') << '\n';

    return 0;
}

