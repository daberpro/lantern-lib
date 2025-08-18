#include "../pch.h"
#include "../Headers/Logging.h"
#include "../Headers/File.h"
#include "../Headers/ModelManager.h"
#include "../ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"
#include "../Dataset/Dataset.h"

#define CNNLayerType lantern::cnn::node::NodeType
#define FFNLayerType lantern::ffn::node::NodeType

int main(int argc, char* argv[])
{

    try
    {
        
        if (argc == 3) {
            
            lantern::modelmanager::ModelManager model_manager(argv[2]);
            lantern::dataset::MnistDataset mnist_dataset;

            if (strcmp(argv[1], "train") == 0) {

                af::info();
                std::cout << '\n';
                af::setSeed(static_cast<unsigned long long>(time(NULL)));

                // Dataset

                // Layer Defintion
                // ==================================================================
                lantern::cnn::layer::Layer CNN_layer; // CNN layer
                lantern::ffn::layer::Layer FFN_layer; // FFN layer

                CNN_layer.SetInputSize({28, 28, 1});
                CNN_layer.AddConvolve(20, af::dim4(0, 0, 0, 0), af::dim4(1, 1), 3, 1);
                CNN_layer.Add<CNNLayerType::RELU>();
                CNN_layer.AddPool<CNNLayerType::AVG_POOL>(2, 2, af::dim4(1, 1));
                CNN_layer.AddConvolve(20, af::dim4(0, 0, 0, 0), af::dim4(1, 1), 7, 20);
                CNN_layer.Add<CNNLayerType::RELU>();
                CNN_layer.AddPool<CNNLayerType::AVG_POOL>(2, 2, af::dim4(1, 1));
                CNN_layer.Add<CNNLayerType::FLATTEN>();
        
                auto* CNN_weights = CNN_layer.GetWeights();
                auto* CNN_bias = CNN_layer.GetBias();
                auto* CNN_prev_gradient = CNN_layer.GetPrevGradient();
                auto* CNN_outputs = CNN_layer.GetOutputs();
                lantern::cnn::optimizer::AdaptiveMomentEstimation CNN_adam;

                lantern::utility::Vector<af::array> FFN_parameters;
                lantern::utility::Vector<af::array> FFN_prev_gradient;
                lantern::utility::Vector<af::array> FFN_outputs;
                lantern::ffn::optimizer::AdaptiveMomentEstimation FFN_adam;
        
                lantern::cnn::feedforward::Initialize(
                    CNN_layer,
                    CNN_adam
                );

        
                FFN_layer.Add<FFNLayerType::SWISH>(CNN_outputs->back().dims(0)); // get input from flatten result of CNN
                FFN_layer.Add<FFNLayerType::LINEAR>(10);
        
                lantern::ffn::feedforward::Initialize(
                    FFN_layer,
                    FFN_parameters,
                    FFN_prev_gradient,
                    FFN_outputs,
                    FFN_adam
                );

                // generate meta data for both model
                CNN_layer.GenerateMetaData();
                FFN_layer.GenerateMetaData();
                
                // ==================================================================

                /**
                 * Training Process
                 */

                const uint32_t batch_size = 100;
                uint32_t total_train_dataset = mnist_dataset.GetTotalTrainImages();
                lantern::utility::Vector<uint32_t> batch_index;
                lantern::utility::Vector<uint32_t> each_class_size = {total_train_dataset};
	            lantern::data::GetRandomSampleClassIndex<batch_size>(
                    batch_index,
                    each_class_size,
                    total_train_dataset
                );
        
                // Create mapping for target hot-encoding output
                af::array mapping_target = af::identity(
                    10,
                    10,
                    f64
                );

                uint32_t epoch = 100, iteration = 0;
                double loss = 1;
                auto image_dims = mnist_dataset.GetTrainImageSizes();
                af::array input, output, target;
                FFN_prev_gradient.push_back(af::array());

                while(iteration < epoch){

                    for(auto& index_data : batch_index){
    
                        input = af::array(
                            image_dims.width,
                            image_dims.height,
                            mnist_dataset.GetTrainImageAt(index_data)
                        );
                        input = input.as(f64);
                        input /= 255.0f;
                        input.eval();
                        target = mapping_target.col(static_cast<int>(*mnist_dataset.GetTrainLabelAt(index_data)));
    
                        CNN_outputs->front() = input;
                        lantern::cnn::feedforward::FeedForward(
                            CNN_layer
                        );
    
                        FFN_outputs.front() = CNN_outputs->back();
                        lantern::ffn::feedforward::FeedForward(
                            FFN_layer,
                            FFN_outputs,
                            FFN_parameters
                        );
    
                
                        output = lantern::probability::SoftMax(FFN_outputs.back());
                        loss = lantern::loss::CrossEntropy(output, target) / static_cast<double>(batch_size);
    
                        std::println("Loss : {}",loss);
                
                        FFN_prev_gradient.back() = lantern::derivative::CrossEntropy(output,target);
                        lantern::ffn::backprop::Backpropagate(
                            FFN_layer,
                            FFN_parameters,
                            FFN_prev_gradient,
                            FFN_outputs,
                            FFN_adam,
                            batch_size   
                        );
    
                        CNN_prev_gradient->back() = af::moddims(
                            FFN_prev_gradient.front(),
                            (*CNN_prev_gradient)[CNN_prev_gradient->size() - 2].dims()
                        );
                        lantern::cnn::backprop::Backpropagate(
                            CNN_layer,
                            CNN_adam,
                            batch_size
                        );
                    }
                    iteration++;

                    lantern::data::GetRandomSampleClassIndex<batch_size>(
                        batch_index,
                        each_class_size,
                        total_train_dataset
                    );
                }

                for (auto& index_data : batch_index) {
                    af::array& CNN_input = CNN_outputs->front();
                    CNN_input = af::array(
                        image_dims.width,
                        image_dims.height,
                        mnist_dataset.GetTrainImageAt(index_data)
                    );
                    CNN_input = CNN_input.as(f64);
                    CNN_input /= 255.0f;
                    lantern::cnn::feedforward::FeedForward(
                        CNN_layer
                    );

                    FFN_outputs.front() = CNN_outputs->back();
                    lantern::ffn::feedforward::FeedForward(
                        FFN_layer,
                        FFN_outputs,
                        FFN_parameters
                    );

                    output = lantern::probability::SoftMax(FFN_outputs.back());
                    mnist_dataset.PrintTrainDataAt(index_data);
                    std::println("Prediction: {}", af::where(af::round(output)));
                }

                af::array& CNN_input = CNN_outputs->front();
                CNN_input = af::array(
                    image_dims.width,
                    image_dims.height,
                    mnist_dataset.GetTrainImageAt(0)
                );
                CNN_input = CNN_input.as(f64);
                CNN_input /= 255.0f;
                lantern::cnn::feedforward::FeedForward(
                    CNN_layer
                );

                FFN_outputs.front() = CNN_outputs->back();
                lantern::ffn::feedforward::FeedForward(
                    FFN_layer,
                    FFN_outputs,
                    FFN_parameters
                );

                output = lantern::probability::SoftMax(FFN_outputs.back());
                mnist_dataset.PrintTrainDataAt(0);
                std::println("Prediction: {}", output);


                // Save model using model manager
                model_manager.Create(); // this is will recreate file if truncate
                model_manager.GetAllData();
                model_manager.AddModel("/CNN", &CNN_layer); // add CNN model to manager
                model_manager.AddModel("/FFN", &FFN_layer); // also add FFN model

                model_manager.SelectModelToModify("/CNN");
                model_manager.AddParams("Weights", *CNN_weights);
                model_manager.AddParams("Bias", *CNN_bias);
                model_manager.SetOutFunctionName(LANTERN_GET_FUNC_NAME(lantern::activation::Linear));

                model_manager.SelectModelToModify("/FFN");
                model_manager.AddParams("Weights_Bias", FFN_parameters);
                model_manager.SetOutFunctionName(LANTERN_GET_FUNC_NAME(lantern::probability::SoftMax));

                std::println("Model was saved to {}",argv[2]);
            }
            else if (strcmp(argv[1],"load") == 0) {

                auto image_dims = mnist_dataset.GetImageSizes();
                lantern::cnn::layer::Layer CNN_layer; // CNN layer
                lantern::ffn::layer::Layer FFN_layer; // FFN layer

                auto* CNN_weights = CNN_layer.GetWeights();
                auto* CNN_bias = CNN_layer.GetBias();
                auto* CNN_outputs = CNN_layer.GetOutputs();

                lantern::utility::Vector<af::array> FFN_parameters;
                lantern::utility::Vector<af::array> FFN_outputs;
        
                model_manager.GetAllData(); // get all data from existed file
                model_manager.LoadModel("/CNN", &CNN_layer);
                model_manager.LoadModel("/FFN", &FFN_layer);

                CNN_layer.PrintLayerInfo();
                FFN_layer.PrintLayerInfo();

                for(uint32_t i = 0; i <= CNN_layer.GetAllLayerSizes()->size(); i++){
                    CNN_outputs->push_back(af::array());
                }

                for(uint32_t i = 0; i < FFN_layer.GetAllLayerSizes()->size(); i++){
                    FFN_outputs.push_back(af::array());
                }

                model_manager.LoadParams("/CNN", "Bias", *CNN_bias);
                model_manager.LoadParams("/CNN", "Weights", *CNN_weights);
                model_manager.LoadParams("/FFN", "Weights_Bias", FFN_parameters);

                std::string CNN_out_func = model_manager.GetOutFuncName("/CNN");
                std::string FFN_out_func = model_manager.GetOutFuncName("/FFN");

                
                af::array& CNN_input = CNN_outputs->front();
                af::array output;

                uint32_t index_input = 0;
                std::string continue_ = "";
                while (true) {

                    std::print("Enter index : ");
                    std::cin >> index_input;
                    std::cin.ignore();
                    
                    if (std::cin.fail()) {
                        std::println("unknown input \"{}\"", index_input);
                        std::cin.clear();
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        continue;
                    }

                    CNN_input = af::array(
                        image_dims.width,
                        image_dims.height,
                        mnist_dataset.GetImageAt(index_input)
                    );
                    CNN_input = CNN_input.as(f64);
                    CNN_input /= 255.0f;
                    lantern::cnn::feedforward::FeedForward(
                        CNN_layer
                    );

                    FFN_outputs.front() = CNN_outputs->back();
                    lantern::ffn::feedforward::FeedForward(
                        FFN_layer,
                        FFN_outputs,
                        FFN_parameters
                    );

                    output = lantern::probability::SoftMax(FFN_outputs.back());
                    mnist_dataset.PrintDataAt(index_input);
                    std::println("Prediction: {}", output);

                    while(true){
                        std::print("Want to coninue? [Y/n]: ");
                        std::cin >> continue_;
                        if(continue_.compare("n") == 0){
                            exit(EXIT_SUCCESS);
                        }else if(continue_.compare("y") != 0){
                            std::println("uknown command \"{}\"",continue_);
                        }else{
                            break;
                        }
                    }


                }

            }
            
        }
        else if (argc == 2) {
            if (strcmp("-h", argv[1]) == 0 || strcmp("--help", argv[1]) == 0) {

                std::println("To use MnistCNN executable file you must specify ");
                std::println("MnistCNN [action] [path]");
                std::println("- action is an action such as \"train\" or \"load\"");
                std::println("- path is the path where model will save or load");

            }
            else {
                std::println("Uknown command, use -h or --help to get help");
            }
        }
        else {
            std::println("Uknown command, use -h or --help to get help");
        }
      
    }
    catch (std::exception &error)
    {

        std::cout << "Error Lantern : " << error.what() << '\n';
        std::cout << "Call stack:\n";
        for (const auto &entry : std::stacktrace::current())
        {
            std::cout << entry << '\n';
        }
        return EXIT_FAILURE;
    }

    return 0;
}