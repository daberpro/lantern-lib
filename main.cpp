#include "pch.h"
#include "Headers/Logging.h"
#include "ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h"
// #include "FeedForwardNetwork/FeedForwardNetwork.h"
// #include "Headers/File.h"
// #include "Dataset/Dataset.h"

#define LayerType lantern::cnn::node::NodeType

int main(){

    af::setSeed(static_cast<unsigned long long>(time(NULL)));

    try{
        
        lantern::cnn::layer::Layer layer;
        layer.AddConvolve(1, 0, 1, 3, 3);
        layer.Add<LayerType::RELU>();
        layer.AddMaxPool(2, 2, 2, 2);
        layer.AddConvolve(4, 0, 1, 2, 1);
        layer.Add<LayerType::RELU>();
        layer.AddMaxPool(2, 2, 2, 2);
        layer.AddConvolve(4, 0, 1, 3, 4);
        layer.Add<LayerType::RELU>();
        layer.AddMaxPool(2, 2, 2, 2);
        layer.Add<LayerType::FLATTEN>();
    
        lantern::utility::Vector<af::array> weights;
        lantern::utility::Vector<af::array> biases;
        lantern::utility::Vector<af::array> prev_gradient;
        lantern::utility::Vector<af::array> outputs;
    
        lantern::optimizer::AdaptiveMomentEstimation adam;

        // lantern::cnn::feedforward::Initialize(
        //     layer,
        //     weights,
        //     biases,
        //     prev_gradient,
        //     outputs,
        //     adam,
        //     {13,13,3}
        // );

        layer.PrintLayerInfo();

        // lantern::utility::Vector<af::array> bout = outputs;
        // outputs[0] = af::randn(13,13,3,f64);
        // lantern::cnn::feedforward::FeedForward(
        //     layer,
        //     weights,
        //     biases,
        //     outputs
        // );

        // lantern::cnn::backprop::Backpropagate(
        //     layer,
        //     weights,
        //     biases,
        //     prev_gradient,
        //     outputs,
        //     adam,
        //     1
        // );

        // for (uint32_t i = 0; i < outputs.size(); i++) {
        //     std::println("First : {} \n Second : {}", bout[i], outputs[i]);
        // }


    }catch(std::exception& error){

        std::cout << "Error Lantern CNN : " << error.what() << '\n';
        return EXIT_FAILURE;

    }
    

	return 0;
}