#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"

int main(int argc, char *argv[])
{
    try
    {

        af::info();
        af::setSeed(static_cast<uint64_t>(std::time(nullptr)));
        af::setBackend(af::Backend::AF_BACKEND_CPU);

        // prepared dataset
        double input_data[] = {
            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0};

        double target_data[] = {
            0.0, 1.0, 1.0, 0.0};

        // wrap dataset into arrayfire
        af::array input = af::array(4, 2, input_data);
        af::array target = af::array(4, 1, target_data);

        lantern::ffn::layer::Layer layer;
        layer.Add<lantern::ffn::node::NodeType::NOTHING>(2);
        layer.Add<lantern::ffn::node::NodeType::SWISH>(6);
        layer.Add<lantern::ffn::node::NodeType::SIGMOID>(1);
        layer.PrintLayerInfo(); // show layer info

        lantern::ffn::optimizer::AdaptiveMomentEstimation optimizer;
        lantern::ffn::feedforward::Initialize(
            layer,
            optimizer
        );

        uint32_t epoch = 500;
        uint32_t iteration = 0;
        uint32_t total_size_of_class = 4;
        const uint32_t batch_size = 4;
        double loss;

        lantern::utility::Vector<uint32_t> batch_index;
        lantern::utility::Vector<uint32_t> each_size = {4};

        lantern::data::GetRandomSampleClassIndex<batch_size>(
            batch_index,
            each_size,
            total_size_of_class
        );
        auto *outputs = layer.GetOutputs();
        auto *prev_gradients = layer.GetPrevradient();

        while (iteration < epoch)
        {

            // get new random index every iter
            lantern::data::GetRandomSampleClassIndex<batch_size>(batch_index, each_size, total_size_of_class);
            for (uint32_t i = 0; i < batch_size; i++)
            {

                outputs->front() = input.row(batch_index[i]).T(); // set the first output as input
                lantern::ffn::feedforward::FeedForward(
                    layer
                );

                loss = lantern::loss::BinaryCrossEntropy(outputs->back(), target.row(batch_index[i]));
                std::println("Loss : {}", loss);

                // get loss gradient to backprop
                prev_gradients->back() = lantern::derivative::BinaryCrossEntropy(outputs->back(), target.row(batch_index[i]));

                lantern::ffn::backprop::Backpropagate(
                    layer,
                    optimizer,
                    batch_size
                );
            }

            iteration++;
        }

        // Show the result training
        af::array predict, actual;
        lantern::data::GetRandomSampleClassIndex<batch_size>(batch_index, each_size, total_size_of_class);
        for (uint32_t i = 0; i < batch_size; i++)
        {

            outputs->front() = input.row(batch_index[i]).T(); // set the first output as input
            lantern::ffn::feedforward::FeedForward(
                layer
            );

            predict = outputs->back();
            actual = target.row(batch_index[i]);

            std::println("Target : {}, Predict : {}", actual, predict);

        }

    }
    catch (std::exception &error)
    {

        std::cout << "\nError Lantern : " << error.what() << "\n\n";
        std::cout << "Call stack:\n";
        for (const auto &entry : std::stacktrace::current())
        {
            std::cout << entry << '\n';
        }
        return EXIT_FAILURE;
    }

    return 0;
}