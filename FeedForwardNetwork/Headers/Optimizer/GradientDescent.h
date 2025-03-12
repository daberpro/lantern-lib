#pragma once
#include "../../Perceptron.h"

namespace lantern {
    namespace perceptron {

        namespace optimizer {
    
            class GradientDescent {
            private:
            
                double learning_rate = 0.01;
                
            public:
                
                #ifdef OPTIMIZE_VERSION
                GradientDescent(double learning_rate): learning_rate(learning_rate) {}
                double GetDelta(const double& gradient, lantern::perceptron::Perceptron* objective, const uint32_t& child_index){
                    return this->learning_rate * gradient;
                }
                #endif

                #ifdef MATRIX_OPTIMIZE
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
                GradientDescent(double learning_rate): learning_rate(learning_rate) {}
                af::array GetDelta(af::array& gradient, int32_t& index){
                    return this->learning_rate * gradient;
                }
                #endif
            };
            
        }
    }
}