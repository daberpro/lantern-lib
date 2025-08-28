#pragma once
#include "FFNBase.h"

namespace lantern
{
    namespace ffn {

        namespace optimizer
        {
    
            class RootMeanSquarePropagation : public Base
            {
            public:
                /**
                 * @brief Construct a new Root Mean Square Propagation Optimizer
                 *
                 * @param learning_rate
                 * @param beta
                 * @param epsilon
                 */
                RootMeanSquarePropagation(const double& _learning_rate = 0.01, const double& _beta_1 = 0.9, const double& _epsilon = 1e-8) : Base(_learning_rate, _beta_1, 0.999, _epsilon) {}
    
                /**
                 * @brief Get the Optimize result of gradient
                 *
                 * @param gradient
                 * @param index
                 * @return af::array
                 */
                af::array GetDelta(const af::array& _gradient, const uint32_t& _index) override
                {
                    this->m_vector_velocity[_index] *= this->m_beta_1;
                    this->m_vector_velocity[_index] += this->m_learning_rate * af::pow(_gradient, 2);
                    this->m_vector_velocity[_index].eval();
                    return this->m_learning_rate * (_gradient / (af::sqrt(this->m_vector_velocity[_index]) + this->m_epsilon));
                }
            };
    
        }
    }
}