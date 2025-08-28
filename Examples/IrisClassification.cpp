#include "../pch.h"
#include "../Headers/File.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"


int main(int argc, char* argv[])
{
    try
    {

        af::info();
        af::setSeed(static_cast<uint64_t>(std::time(nullptr)));
        std::cout << '\n';

        // Read CSV File from Iris Flower Dataset
        std::string current_path = std::filesystem::current_path().string();
        // Create partition data and split data into 80% train 20% test
        auto dataset = lantern::file::ReadCSVFile(current_path + "/dataset/Iris_Flower/IRIS.csv");
        auto [train, test] = lantern::data::PartitionDataset(dataset, 0.8, true);

        // container to hold train input data and target data
        lantern::utility::Vector<double> train_input_data;
        lantern::utility::Vector<uint32_t> train_target_data;

        // container to hold test input data and target data
        lantern::utility::Vector<double> test_input_data;
        lantern::utility::Vector<uint32_t> test_target_data;

        // get index map to manually mapping target string 
        auto* dataset_index_map = dataset.GetIndexMapPtr();
        dataset_index_map->insert({ "Iris-setosa",       0 }); // set value for Iris-setosa to become index 0
        dataset_index_map->insert({ "Iris-versicolor",   1 }); // set value for Iris-setosa to become index 1
        dataset_index_map->insert({ "Iris-virginica",    2 }); // set value for Iris-setosa to become index 2

        // create definition of FFN layer
        lantern::ffn::layer::Layer layer;
        layer.Add<lantern::ffn::node::NodeType::NOTHING>(4); // set first node as an input (NOTHING activation)
        layer.Add<lantern::ffn::node::NodeType::RELU>(20);
        layer.Add<lantern::ffn::node::NodeType::RELU>(10);
        layer.Add<lantern::ffn::node::NodeType::LINEAR>(3); // the last output must be 3 same as the target class because we use SOFTMAX
        layer.PrintLayerInfo(); // show layer info

        // use ADAM optimizer with default value
        lantern::ffn::optimizer::AdaptiveMomentEstimation optimizer;
        lantern::ffn::feedforward::Initialize(
            layer,
            optimizer
        );

        uint32_t epoch = 500;
        uint32_t iteration = 0;
        double loss;
        af::array input, target;
        auto* outputs = layer.GetOutputs();
        auto* prev_gradients = layer.GetPrevradient();

        uint32_t total_size_of_train_class = train.size();
        uint32_t train_batch_size = total_size_of_train_class;
        lantern::utility::Vector<uint32_t> train_batch_index;
        lantern::utility::Vector<uint32_t> train_each_size = { train_batch_size };

        // Get random index from train dataset
        lantern::data::GetRandomSampleClassIndex(
            train_batch_size,
            train_batch_index,
            train_each_size,
            total_size_of_train_class,
            true
        );

        // create mapping for target to be one hot encode
        lantern::utility::Vector<double> target_map = {
            1,0,0,
            0,1,0,
            0,0,1
        };
        af::array target_map_array = af::array(3, 3, target_map.data());

        while (iteration < epoch)
        {

            // get new random index every iter
            lantern::data::GetRandomSampleClassIndex(
                train_batch_size,
                train_batch_index,
                train_each_size,
                total_size_of_train_class,
                true
            );
            for (uint32_t i = 0; i < train_batch_size; i++)
            {


                dataset.Row<double>(train_input_data, train, train_batch_index[i], 0, 4);
                dataset.RowIndexMapping(train_target_data, train, train_batch_index[i], 4, 1);

                input = af::array(1, 4, train_input_data.data());
                target = target_map_array.row(train_target_data.front());

                outputs->front() = input.T(); // set the first output as input
                lantern::ffn::feedforward::FeedForward(
                    layer
                );

                af::array output = lantern::probability::SoftMax(outputs->back());
                loss = lantern::loss::CrossEntropy(output, target.T());
                std::println("Loss : {}", loss);

                // get loss gradient to backprop
                prev_gradients->back() = lantern::derivative::CrossEntropy(output, target.T());

                lantern::ffn::backprop::Backpropagate(
                    layer,
                    optimizer,
                    train_batch_size
                );
            }

            iteration++;
        }

        // Show the result training
        af::array predict, actual;
        uint32_t total_size_of_test_class = test.size();
        uint32_t test_batch_size = total_size_of_test_class;
        lantern::utility::Vector<uint32_t> test_batch_index;
        lantern::utility::Vector<uint32_t> test_each_size = { test_batch_size };

        lantern::data::GetRandomSampleClassIndex(test_batch_size, test_batch_index, test_each_size, total_size_of_test_class);
        for (uint32_t i = 0; i < test_batch_size; i++)
        {

            dataset.Row<double>(test_input_data, test, test_batch_index[i], 0, 4);
            dataset.RowIndexMapping(test_target_data, test, test_batch_index[i], 4, 1);
            input = af::array(1, 4, test_input_data.data());

            outputs->front() = input.T();
            lantern::ffn::feedforward::FeedForward(
                layer
            );

            predict = lantern::probability::SoftMax(outputs->back());
            actual = target_map_array.row(test_target_data.front());

            std::println("Target : {}, Predict : {}", actual, predict.T());

        }

    }
    catch (std::exception& error)
    {

        std::cout << "\nError Lantern : " << error.what() << "\n\n";
        std::cout << "Call stack:\n";
        for (const auto& entry : std::stacktrace::current())
        {
            std::cout << entry << '\n';
        }
        return EXIT_FAILURE;
    }

    return 0;
}