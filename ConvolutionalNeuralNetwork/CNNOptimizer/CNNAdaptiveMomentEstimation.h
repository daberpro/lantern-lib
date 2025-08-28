#pragma once
#include "CNNBase.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {

            class AdaptiveMomentEstimation : public Base {
            public:
                /**
                 * @brief Construct a new Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 */
                AdaptiveMomentEstimation(const double& _learning_rate = 0.01f): Base(_learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& _gradient_w, const af::array& _gradient_b,const uint32_t& _index) override {
                    this->m_iteration++;
    
                    this->m_w_stack_previous_gradient[_index] = 
                        this->m_beta_1 * this->m_w_stack_previous_gradient[_index] + 
                        (1.0 - this->m_beta_1) * _gradient_w;
    
                    this->m_w_vector_velocity[_index] = 
                        this->m_beta_2 * this->m_w_vector_velocity[_index] + 
                        (1.0 - this->m_beta_2) * af::pow(_gradient_w, 2);
    
                    this->m_w_mt = this->m_w_stack_previous_gradient[_index] / 
                        (1.0 - std::pow(this->m_beta_1, this->m_iteration));
                    
                    this->m_w_vt = this->m_w_vector_velocity[_index] / 
                        (1.0 - std::pow(this->m_beta_2, this->m_iteration));
    
                    this->m_w_stack_previous_gradient[_index].eval();
                    this->m_w_vector_velocity[_index].eval();
                    this->m_w_mt.eval();
                    this->m_w_vt.eval();

                    this->m_b_stack_previous_gradient[_index] = 
                        this->m_beta_1 * this->m_b_stack_previous_gradient[_index] + 
                        (1.0 - this->m_beta_1) * _gradient_b;
    
                    this->m_b_vector_velocity[_index] = 
                        this->m_beta_2 * this->m_b_vector_velocity[_index] + 
                        (1.0 - this->m_beta_2) * af::pow(_gradient_b, 2);
    
                    this->m_b_mt = this->m_b_stack_previous_gradient[_index] / 
                        (1.0 - std::pow(this->m_beta_1, this->m_iteration));
                    
                    this->m_b_vt = this->m_b_vector_velocity[_index] / 
                        (1.0 - std::pow(this->m_beta_2, this->m_iteration));
    
                    this->m_b_stack_previous_gradient[_index].eval();
                    this->m_b_vector_velocity[_index].eval();
                    this->m_b_mt.eval();
                    this->m_b_vt.eval();
    
                    return {
                        (this->m_learning_rate * this->m_w_mt) / (af::sqrt(this->m_w_vt) + this->m_epsilon),
                        (this->m_learning_rate * this->m_b_mt) / (af::sqrt(this->m_b_vt) + this->m_epsilon)
                    };
                }

                /**
                 * @brief Get the Optimize result of other parameters
                 * 
                 * @param batch_norm_gradient 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDeltaBatchNorm(const af::array& batch_norm_gradient, const uint32_t& index) override {
                    this->m_batch_norm_stack_previous_gradient[index] = 
                        this->m_beta_1 * this->m_batch_norm_stack_previous_gradient[index] + 
                        (1.0 - this->m_beta_1) * batch_norm_gradient;
    
                    this->m_batch_norm_vector_velocity[index] = 
                        this->m_beta_2 * this->m_batch_norm_vector_velocity[index] + 
                        (1.0 - this->m_beta_2) * af::pow(batch_norm_gradient, 2);
    
                    this->m_batch_norm_mt = this->m_batch_norm_stack_previous_gradient[index] / 
                        (1.0 - std::pow(this->m_beta_1, this->m_iteration));
                    
                    this->m_batch_norm_vt = this->m_batch_norm_vector_velocity[index] / 
                        (1.0 - std::pow(this->m_beta_2, this->m_iteration));
    
                    this->m_batch_norm_stack_previous_gradient[index].eval();
                    this->m_batch_norm_vector_velocity[index].eval();
                    this->m_batch_norm_mt.eval();
                    this->m_batch_norm_vt.eval();
                    
                    return (this->m_learning_rate * this->m_batch_norm_mt) / (af::sqrt(this->m_batch_norm_vt) + this->m_epsilon);
                }
            };

        }
    }
}