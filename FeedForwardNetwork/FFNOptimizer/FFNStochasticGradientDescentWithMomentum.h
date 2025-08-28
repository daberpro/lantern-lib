#pragma once
#include "FFNBase.h"

namespace lantern
{
    namespace ffn {

        namespace optimizer
        {
    
            class StochasticGradientDescentWithMomentum : public Base
            {
            public: 
                /**
                 * @brief Construct a new Stochastic Gradient Descent With Momentum Optimizer
                 *
                 * @param learning_rate
                 * @param beta
                 */
                StochasticGradientDescentWithMomentum(const double& _learning_rate = 0.01, const double& _beta_1 = 0.9) : Base(_learning_rate,_beta_1) {}
                
                /**
                 * @brief Get the Optimize result of _gradient
                 *
                 * @param _gradient
                 * @param _index
                 * @return af::array
                 */
                af::array GetDelta(const af::array& _gradient, const uint32_t& _index)
                {
                    this->m_vector_velocity[_index] *= this->m_beta_1;
                    this->m_vector_velocity[_index] += this->m_learning_rate * _gradient;
                    this->m_vector_velocity[_index].eval();
                    return this->m_vector_velocity[_index];
                }
            };
    
        }
    }
}