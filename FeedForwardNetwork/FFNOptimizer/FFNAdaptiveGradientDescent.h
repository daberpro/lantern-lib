#pragma once
#include "FFNBase.h"

namespace lantern
{
    namespace ffn {

        namespace optimizer
        {
    
            class AdaptiveGradientDescent : public Base
            {
            public:
                /**
                 * @brief Construct a new Adaptive Gradient Descent Optimizer
                 *
                 * @param learning_rate
                 * @param epsilon
                 */
                AdaptiveGradientDescent(double learning_rate = 0.01, double epsilon = 1e-08) : Base(learning_rate, 0.9, 0.999, epsilon) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 *
                 * @param gradient
                 * @param index
                 * @return af::array
                 */
                af::array GetDelta(af::array &gradient, uint32_t &index) override
                {
                    this->stack_previous_gradient[index] += af::pow(gradient, 2);
                    this->stack_previous_gradient[index].eval();
                    return (this->learning_rate / (af::sqrt(this->stack_previous_gradient[index]) + this->epsilon)) * gradient;
                }
            };
    
        }
    }
}