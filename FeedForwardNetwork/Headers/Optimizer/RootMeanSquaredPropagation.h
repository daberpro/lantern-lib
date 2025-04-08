#pragma once
#include "../../Perceptron.h"

namespace lantern
{
    namespace perceptron
    {

        namespace optimizer
        {

            class RootMeanSquarePropagation
            {
            private:
                double learning_rate = 0.01, beta = 0.9, epsilon = 1e-8;
                
            public:
            
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
                /**
                 * @brief Construct a new Root Mean Square Propagation Optimizer
                 * 
                 * @param learning_rate 
                 * @param beta 
                 * @param epsilon 
                 */
                RootMeanSquarePropagation(double learning_rate = 0.01, double beta = 0.9,double epsilon = 1e-8) : learning_rate(learning_rate), beta(beta), epsilon(epsilon) {}
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDelta(af::array& gradient, int32_t& index)
                {
                    this->vector_velocity[index] *= this->beta;
                    this->vector_velocity[index] += this->learning_rate * pow(gradient,2);
                    this->vector_velocity[index].eval();
                    return this->learning_rate * (gradient/(af::sqrt(this->vector_velocity[index])+this->epsilon));
                }

            };

        }
    }
}