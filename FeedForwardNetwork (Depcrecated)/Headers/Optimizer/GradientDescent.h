#pragma once
#include "../../Perceptron.h"

namespace lantern {
    namespace perceptron {

        namespace optimizer {
    
            class GradientDescent {
            private:
            
                double learning_rate = 0.01;
                
            public:
                
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
                /**
                 * @brief Construct a new Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 */
                GradientDescent(double learning_rate = 0.01f): learning_rate(learning_rate) {}
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDelta(af::array& gradient, int32_t& index){
                    return this->learning_rate * gradient;
                }
            };
            
        }
    }
}