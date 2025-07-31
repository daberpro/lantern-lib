#include "pch.h"
#include "Headers/Logging.h"
#include "ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h"
// #include "FeedForwardNetwork/FeedForwardNetwork.h"
// #include "Headers/File.h"
// #include "Dataset/Dataset.h"

#define LayerType lantern::cnn::node::NodeType

int main()
{

    af::info();
    std::cout << '\n';
    af::setSeed(static_cast<unsigned long long>(time(NULL)));

    try
    {

        lantern::cnn::layer::Layer layer;
        layer.SetInputSize({16, 16, 3});

        /**
         * =============================================================
         */
        layer.AddConvolve(
            1,
            af::dim4(0, 0, 0, 0),
            af::dim4(1, 1),
            3,
            3
        );
        /**
         * Out width -> floor((input_width - kernel_size + 2 * padding_x)/stride_w + 1) * total_node
         * Out width -> floor((input_height - kernel_size + 2 * padding_y)/stride_h + 1) * total_node
         * 
         * Out width -> floor((16 - 3 + 2 * 0)/1 + 1) * 1 = 6
         * Out height -> floor((16 - 3 + 2 * 0)/1 + 1) * 1 = 6
         * 
         */
        layer.Add<LayerType::SWISH>();
        layer.AddPool<LayerType::MAX_POOL>(
            2, 
            2, 
            af::dim4(2, 2)
        );
        /**
         * Out width -> ((prev_out_width / stride_w) - (prev_out_width / stride_w)  % stride_w) / pool_w
         * Out height -> ((prev_out_height / stride_h) - (prev_out_height / stride_h) % stride_h) / poll_h
         *
         * Out width -> ((6 / 2) - 1) / 2 = 1
         * Out height -> ((6 / 2) - 1) / 2 = 1
         * /
        /**
         * =============================================================
         */
        layer.AddConvolve(
            4, 
            af::dim4(0, 0, 0, 0), 
            af::dim4(1, 1), 
            2, 
            1
        );
        layer.Add<LayerType::SWISH>();
        layer.AddPool<LayerType::AVG_POOL>(
            2, 
            2, 
            af::dim4(1, 1)
        );
        /**
         * =============================================================
         */
        // layer.AddConvolve(4, af::dim4(0, 0, 0, 0), af::dim4(1, 1), 3, 4);
        // layer.Add<LayerType::SWISH>();
        // layer.AddPool<LayerType::AVG_POOL>(2, 2, af::dim4(1, 1));
        layer.Add<LayerType::FLATTEN>();

        lantern::utility::Vector<af::array> weights;
        lantern::utility::Vector<af::array> bias;
        lantern::utility::Vector<af::array> prev_gradient;
        lantern::utility::Vector<af::array> outputs;

        lantern::cnn::optimizer::GradientDescent gd;

        lantern::experimental::cnn::feedforward::Initialize(
            layer,
            weights,
            bias,
            prev_gradient,
            outputs,
            gd
        );

        layer.PrintLayerInfo();

        outputs[0] = af::randn(16,16,3,f64);
        lantern::experimental::cnn::feedforward::FeedForward(
            layer,
            weights,
            bias,
            outputs
        );

        std::println("{}",outputs);

        prev_gradient.back() = af::constant(1.0f, outputs.back().dims(), f64);
        lantern::cnn::backprop::Backpropagate(
             layer,
             weights,
             bias,
             prev_gradient,
             outputs,
             gd,
             1
         );

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
    }
    catch (std::exception &error)
    {

        std::cout << "Error Lantern CNN : " << error.what() << '\n';
        std::cout << "Call stack:\n";
        for (const auto &entry : std::stacktrace::current())
        {
            std::cout << entry << '\n';
        }
        return EXIT_FAILURE;
    }

    return 0;
}