#pragma once
#include "Perceptron.h"
#include "Layering.h"
#include "ActivationFunction.h"
#include "PerceptronFeedForward.h"
#include "BackPropagation.h"
#include "LossFunction.h"

namespace lantern {
    
    class FeedForwardNetwork{
    private:
        uint32_t layer = 0;
        lantern::utility::Vector<lantern::perceptron::Perceptron> inputs;
        lantern::utility::Vector<lantern::utility::Vector<lantern::perceptron::Perceptron>> hiddens;
        lantern::utility::Vector<lantern::perceptron::Perceptron> outputs;

        lantern::utility::Vector<af::array> network_parameters;
    	lantern::utility::Vector<af::array> network_gradient_based_parameters;
        lantern::utility::Vector<af::array> network_outputs;
        lantern::utility::Vector<lantern::perceptron::Activation> network_operators;

    public:

        template <lantern::perceptron::Activation activation>
        void AddInputLayer(uint32_t total_perceptron){
            if(this->layer == 0){
                for(uint32_t i = 0; i < total_perceptron; i++){
                    this->inputs.push_back(
                        lantern::perceptron::Perceptron("i" + std::to_string(i))
                    );
                }
                return;
            }
            std::cerr << "Lantern Feed Forward Network Error, cannot add input layer after other type of layer\n";
            std::cerr << "input layer must be call first before other layer\n";
            exit(EXIT_FAILURE);
        };

        template <lantern::perceptron::Activation activation>
        void AddHiddenLayer(uint32_t total_perceptron){
            lantern::utility::Vector<lantern::perceptron::Perceptron*> parents;
            if(this->layer == 0){
                for(auto& p : this->inputs){
                    parents.push_back(&p);
                }
            }else{
                lantern::utility::Vector<lantern::perceptron::Perceptron>& l = this->hiddens[this->layer-1];
                for(auto& p : l){
                    parents.push_back(&p);
                }
            }
            
            this->layer++;
            lantern::utility::Vector<lantern::perceptron::Perceptron> temp;
            
            for(uint32_t i = 0; i < total_perceptron; i++){
                lantern::perceptron::Perceptron h("h"+std::to_string(this->layer)+"_"+std::to_string(i));
                h.layer = this->layer;
                h.op = activation;
                h.parents.copyPtrData(parents);
                temp.push_back(
                    std::move(h)
                );
            }

            this->hiddens.push_back(std::move(temp));

        };

        template <lantern::perceptron::Activation activation>
        void AddOutputLayer(uint32_t total_perceptron){
            lantern::utility::Vector<lantern::perceptron::Perceptron*> parents;
            lantern::utility::Vector<lantern::perceptron::Perceptron>& l = this->hiddens[this->layer-1];
            for(auto& p : l){
                parents.push_back(&p);
            }
            
            this->layer++;
            for(uint32_t i = 0; i < total_perceptron; i++){
                lantern::perceptron::Perceptron o("o"+std::to_string(this->layer)+"_"+std::to_string(i));
                o.layer = this->layer;
                o.op = activation;
                o.parents.copyPtrData(parents);
                this->outputs.push_back(
                    std::move(o)
                );
            }
        };

        template <typename Optimizer>
        void Train(Optimizer optimizer){
            // lantern::perceptron::FeedForward(
            //     &o1,
            //     this->network_parameters,
            //     this->network_gradient_based_parameters, 
            //     this->network_operators,
            //     this->network_outputs,
            //     optimizer
            // );
        };
        void Predict();

        friend void print(lantern::FeedForwardNetwork& ffn);

    };

}

