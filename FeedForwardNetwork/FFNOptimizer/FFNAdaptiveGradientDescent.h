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
                AdaptiveGradientDescent(const double& _learning_rate = 0.01, const double& _epsilon = 1e-08) : Base(_learning_rate, 0.9, 0.999, _epsilon) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 *
                 * @param gradient
                 * @param _index
                 * @return af::array
                 */
                af::array GetDelta(const af::array& _gradient, const uint32_t& _index) override
                {
                    this->m_stack_previous_gradient[_index] += af::pow(_gradient, 2);
                    this->m_stack_previous_gradient[_index].eval();
                    return (this->m_learning_rate / (af::sqrt(this->m_stack_previous_gradient[_index]) + this->m_epsilon)) * _gradient;
                }
            };
    
        }
    }
}