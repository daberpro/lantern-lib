#include "pch.h"
#include "Headers/Logging.h"
#include "Headers/File.h"
#include "Headers/ModelManager.h"
#include "Headers/Utility.h"
#include "ConvolutionalNeuralNetwork/ConvolutionalNeuralNetwork.h"
#include "FeedForwardNetwork/FeedForwardNetwork.h"
#include "Dataset/Dataset.h"

#define CNNLayerType lantern::cnn::node::NodeType
#define FFNLayerType lantern::ffn::node::NodeType

int main(int argc, char* argv[])
{
    try
    {
        
        if (argc == 3) {

            /*af::setBackend(af::Backend::AF_BACKEND_CUDA);*/
            
            const uint32_t IMAGE_WIDTH = 200;
            const uint32_t IMAGE_HEIGHT = 200;
            const uint32_t BATCH_SIZE = 100;
            lantern::modelmanager::ModelManager model_manager(argv[2]);

            if (strcmp(argv[1], "train") == 0) {

                af::info();
                std::cout << '\n';
                af::setSeed(static_cast<unsigned long long>(time(NULL)));

                lantern::dataset::ImageLoader<BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, true> image_dataset;
                // Dataset
                std::string _current_path = std::filesystem::current_path().string();
                std::string _training_path = "/cat_and_dog/training_set/training_set/";
                image_dataset.CreateDatasetForFolder("TrainDataset");
                image_dataset.SelectDatasetToModify("TrainDataset");
                image_dataset.GetImagesDataFromFolder(_current_path + _training_path + "/cats");
                image_dataset.GetImagesDataFromFolder(_current_path + _training_path + "/dogs");
                image_dataset.ReadCSVLabelDataFromFolder(_current_path + _training_path + "/train_labels.csv");

                // Layer Defintion
                // ==================================================================
                lantern::cnn::layer::Layer CNN_layer; // CNN layer
                lantern::ffn::layer::Layer FFN_layer; // FFN layer

                auto* CNN_outputs = CNN_layer.GetOutputs();
                auto* CNN_prev_gradient = CNN_layer.GetPrevGradient();
                auto* CNN_weights = CNN_layer.GetWeights();
                auto* CNN_bias = CNN_layer.GetBias();

                // Set input image to be 200 x 200 x 3 (RGB)
                CNN_layer.SetInputSize({IMAGE_WIDTH, IMAGE_HEIGHT, 3});
                CNN_layer.AddConvolve(
                    20,
                    af::dim4((dim_t)0, (dim_t)0, (dim_t)0, (dim_t)0),
                    af::dim4(1, 1),
                    3,
                    3
                );
                CNN_layer.Add<CNNLayerType::RELU>();
                CNN_layer.AddPool<CNNLayerType::AVG_POOL>(
                    2,
                    2,
                    af::dim4(1, 1)
                );
                CNN_layer.AddConvolve(
                    20,
                    af::dim4((dim_t)0, (dim_t)0, (dim_t)0, (dim_t)0),
                    af::dim4(1, 1),
                    7,
                    20
                );
                CNN_layer.Add<CNNLayerType::RELU>();
                CNN_layer.AddPool<CNNLayerType::AVG_POOL>(
                    2,
                    2,
                    af::dim4(1, 1)
                );
                
                CNN_layer.Add<CNNLayerType::FLATTEN>();
        
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
                FFN_layer.Add<FFNLayerType::SWISH>(2); // get input from flatten result of CNN
                FFN_layer.Add<FFNLayerType::SIGMOID>(1);
        
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

                uint32_t epoch = 10, iteration = 0;
                double loss = 1;
                af::array input, output, target;
                FFN_prev_gradient.push_back(af::array());
                image_dataset.Run();

                lantern::utility::Vector<af::array> input_images;
                lantern::utility::Vector<std::string> label_images;
                auto csv_label = image_dataset.GetCSVFile().RowsWithHeader<double>();

                for(uint32_t i = 0; i < BATCH_SIZE; i ++){
                    input_images.push_back(af::array());
                    label_images.push_back("");
                }

                while(iteration < epoch){

                    for (uint32_t i = 0; i < BATCH_SIZE; i++) {

                        image_dataset.GetAsAF<f64>(input_images[i], label_images[i]);
                        target = af::array(1, 1, csv_label[label_images[i]].getData());

                        CNN_outputs->front() = input_images[i];
                        lantern::cnn::feedforward::FeedForward(
                            CNN_layer
                        );
    
                        FFN_outputs.front() = CNN_outputs->back();
                        lantern::ffn::feedforward::FeedForward(
                            FFN_layer,
                            FFN_outputs,
                            FFN_parameters
                        );
    
                        output = FFN_outputs.back();
                        loss = lantern::loss::BinaryCrossEntropy(output, target);
    
                        std::println("Loss : {}",loss);
                
                        FFN_prev_gradient.back() = lantern::derivative::BinaryCrossEntropy(output,target);
                        lantern::ffn::backprop::Backpropagate(
                            FFN_layer,
                            FFN_parameters,
                            FFN_prev_gradient,
                            FFN_outputs,
                            FFN_adam,
                            BATCH_SIZE
                        );
    
                        CNN_prev_gradient->back() = af::moddims(
                            FFN_prev_gradient.front(),
                            (*CNN_prev_gradient)[CNN_prev_gradient->size() - 2].dims()
                        );
                        lantern::cnn::backprop::Backpropagate(
                            CNN_layer,
                            CNN_adam,
                            BATCH_SIZE
                        );

                        
                    }
                    iteration++;
                }

                image_dataset.Stop();

                
                // for (uint32_t i = 0; i < BATCH_SIZE; i++) {
                //     auto data = image_dataset.GetImageData();
                //     auto label = data.second;

                //     input = data.first.as(f64);
                //     input /= 255.0f;
                //     input.eval();
                //     input = af::resize(input, IMAGE_WIDTH, IMAGE_HEIGHT, AF_INTERP_NEAREST);
                //     target = af::constant(lantern::utility::ConvertFromString<double>((*label)[1]), 1, 1);

                //     CNN_outputs.front() = input;
                //     lantern::cnn::feedforward::FeedForward(
                //         CNN_layer,
                //         CNN_weights,
                //         CNN_bias,
                //         CNN_outputs
                //     );

                //     FFN_outputs.front() = CNN_outputs.back();
                //     lantern::ffn::feedforward::FeedForward(
                //         FFN_layer,
                //         FFN_outputs,
                //         FFN_parameters
                //     );
                //     output = FFN_outputs.back();
                //     std::println("Prediction : {}, Target {}", output, target);
                // }



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
                model_manager.SetOutFunctionName(LANTERN_GET_FUNC_NAME(lantern::activation::Linear));

                

                std::println("Model was saved to {}",argv[2]);
            }
            else if (strcmp(argv[1],"load") == 0) {
                


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