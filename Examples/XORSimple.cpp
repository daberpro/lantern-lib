#include "../pch.h"
#include "../Headers/Logging.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"

std::string to_lower(std::string str){
    std::transform(
        str.begin(),
        str.end(),
        str.begin(),
        [](uint8_t char_){
            return std::tolower(char_);
        }
    );

    return str;
}

int main(int argc, char* argv[]){

    af::setSeed(static_cast<uint64_t>(std::time(nullptr)));
    
    double input_data[] = {
        1.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.0
    };

    double target_data[] = {
        0.0, 1.0, 1.0, 0.0
    };

    af::array input = af::array(4,2,input_data);
    af::array target = af::array(4,1,target_data);

    if(argc == 3){
        
        // check if train or load model
        if(strcmp("train", to_lower(argv[1]).c_str()) == 0){


            lantern::ffn::layer::Layer layer;
            layer.Add<lantern::ffn::node::NodeType::NOTHING>(2);
            layer.Add<lantern::ffn::node::NodeType::SWISH>(4);
            layer.Add<lantern::ffn::node::NodeType::SWISH>(6);
            layer.Add<lantern::ffn::node::NodeType::SIGMOID>(1);

            lantern::ffn::optimizer::AdaptiveMomentEstimation adam;

            lantern::feedforward::FeedForwardNetwork model(
                &input,
                &target,
                &layer,
                {4},
                1e-08,
                200
            );

            model.Train<4>(
                adam,
                lantern::loss::SumSquareResidual,
                lantern::derivative::SumSquareResidual,
                lantern::activation::Linear
            );

            std::cout << "Prediction Result:\n";
            af::array result;
            model.Predict(
                input,
                result,
                lantern::activation::Linear
            );
            std::cout << result << '\n';

            model.SaveModel(
                argv[2],
                LANTERN_GET_FUNC_NAME(lantern::activation::Linear)
            );

            return EXIT_SUCCESS;

            
        }else if(strcmp("load", to_lower(argv[1]).c_str()) == 0){

            lantern::feedforward::FeedForwardNetwork model;
            model.LoadModel(argv[2]);
            std::cout << "Prediction Result:\n";
            af::array result;
            model.Predict(
                input,
                result,
                lantern::activation::Linear
            );
            std::cout << result << '\n';

            return EXIT_SUCCESS;

        }else{

            std::cout << "Unknown params, type -h or --help to show help\n";
            return EXIT_SUCCESS;

        }

    }
    else if (argc == 2) {
        if (strcmp("-h", to_lower(argv[1]).c_str()) == 0 || strcmp("--help", to_lower(argv[1]).c_str()) == 0) {

            std::cout << "To use XORSimple executable file you must specify \n";
            std::cout << "XORSimple [action] [path]\n";
            std::cout << "- action is an action such as \"train\" or \"load\"\n";
            std::cout << "- path is the path where model will save or load\n";

            return EXIT_SUCCESS;

        }
    }
    
    std::cout << "Unknown params, type -h or --help to show help\n";
    return EXIT_SUCCESS;

}