#pragma once
#include "../../Perceptron.h"

namespace latern {
    namespace perceptron {

        namespace optimizer {
    
            class GradientDescent {
            private:
            
                double learning_rate = 0.01;
                uint32_t iteration = 0;
            
            public:
            
                GradientDescent(double learning_rate): learning_rate(learning_rate) {}
                double GetDelta(const double& gradient, latern::perceptron::Perceptron* objective, const uint32_t& child_index){
                    return this->learning_rate * gradient;
                }

                void SetIteration(const uint32_t& iter){
                    this->iteration = iter;
                }

                uint32_t GetIteration(){
                    return this->iteration;
                }
            
            };
            
        }
    }
}