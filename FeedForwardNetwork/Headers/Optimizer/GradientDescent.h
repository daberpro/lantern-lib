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
                
                #ifdef OPTIMIZE_VERSION
                GradientDescent(double learning_rate): learning_rate(learning_rate) {}
                double GetDelta(const double& gradient, latern::perceptron::Perceptron* objective, const uint32_t& child_index){
                    return this->learning_rate * gradient;
                }
                #endif

                #ifdef MATRIX_OPTIMIZE
                GradientDescent(double learning_rate): learning_rate(learning_rate) {}
                af::array GetDelta(af::array& gradient){
                    return this->learning_rate * gradient;
                }
                #endif

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