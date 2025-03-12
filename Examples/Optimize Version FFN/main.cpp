#include "../../pch.h"
#define OPTIMIZE_VERSION
#include "../../Headers/Vector.h"
#include "../../FeedForwardNetwork/FeedForwardNetwork.h"

int main(){

    lantern::perceptron::Perceptron i1(1.0f,"i1");
    lantern::perceptron::Perceptron i2(1.0f,"i2");
    lantern::perceptron::Perceptron h1("h1");
    lantern::perceptron::Perceptron h2("h2");
    lantern::perceptron::Perceptron h3("h3");
    lantern::perceptron::Perceptron h4("h4");
    lantern::perceptron::Perceptron o1("o1");

    lantern::perceptron::activation::Swish(h1,i1,i2);
    lantern::perceptron::activation::Swish(h2,i1,i2);
    lantern::perceptron::activation::Swish(h3,i1,i2);
    lantern::perceptron::activation::Swish(h4,i1,i2);
    lantern::perceptron::activation::Sigmoid(o1,h1,h2,h3,h4);

    double inputs[4][2] = {
        {1,1},
        {1,0},
        {0,1},
        {0,0}
    };

    double outputs[4] = {
        0,
        1,
        1,
        0
    };

    lantern::utility::Vector<lantern::perceptron::Perceptron*> fix_position_nodes;
    lantern::perceptron::FeedForward(
        &o1,
        fix_position_nodes
    );

    uint32_t iter = 0, selected_index = 0;
    double loss = 0;
    lantern::perceptron::optimizer::AdaptiveMomentEstimation adam;
    // lantern::perceptron::optimizer::GradientDescent gd;

    std::random_device rd;
    std::mt19937 rg(rd());
    std::uniform_int_distribution<> rand(0,3);

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    while(true){

        selected_index = rand(rg);
        i1.value = inputs[selected_index][0];
        i2.value = inputs[selected_index][1];
        lantern::perceptron::FeedForward(fix_position_nodes);  
        loss = pow(outputs[selected_index] - o1.value,2);

        std::cout << "Inputs: ["        << inputs[selected_index][0] 
                  << ","                << inputs[selected_index][1] 
                  << "] | Predicted : " << o1.value 
                  << ", Target : "      << outputs[selected_index] 
                  << " | Loss : "       << loss 
                  << "\n";

        o1.gradient_based_input[0] = -2 * (outputs[selected_index] - o1.value);
        
        lantern::perceptron::BackPropagation(
            o1,
            adam
        );

        if(loss <= 0.001 && iter % 100 == 0){
            break;
        }

        iter++;

    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_running_times = end - start;
    std::cout << std::string(70,'=') << '\n';
    std::cout << "Total time : " << total_running_times.count() << "s\n";
    std::cout << std::string(70,'=') << '\n';


    return 0;
}