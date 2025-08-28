#pragma once
#include "CNNBase.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {

            class StochasticGradientDescentWithMomentum : public Base {
            public:
                /**
                 * @brief Construct a new Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 */
                StochasticGradientDescentWithMomentum(const double& _learning_rate = 0.01f): Base(_learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& _gradient_w, const af::array& _gradient_b,const uint32_t& _index) override {
                    //for weights
                    this->m_w_vector_velocity[_index] *= this->m_beta_1;
                    this->m_w_vector_velocity[_index] += this->m_learning_rate * _gradient_w;
                    this->m_w_vector_velocity[_index].eval();

                    // for bias
                    this->m_b_vector_velocity[_index] *= this->m_beta_1;
                    this->m_b_vector_velocity[_index] += this->m_learning_rate * _gradient_b;
                    this->m_b_vector_velocity[_index].eval();

                    return {
                        this->m_w_vector_velocity[_index],
                        this->m_b_vector_velocity[_index]
                    };
                }

                /**
                 * @brief Get the Optimize result of other parameters
                 * 
                 * @param batch_norm_gradient 
                 * @param _index 
                 * @return af::array 
                 */
                af::array GetDeltaBatchNorm(const af::array& _batch_norm_gradient, const uint32_t& _index) override {
                    // for other parameters
                    this->m_batch_norm_vector_velocity[_index] *= this->m_beta_1;
                    this->m_batch_norm_vector_velocity[_index] += this->m_learning_rate * _batch_norm_gradient;
                    this->m_batch_norm_vector_velocity[_index].eval();
                    
                    return this->m_batch_norm_vector_velocity[_index];
                }
            };

        }
    }
}