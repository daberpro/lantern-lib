#pragma once
#include "FFNBase.h"

namespace lantern {
    namespace ffn {

        namespace optimizer {
            class AdaptiveMomentEstimation : public Base {
            public:
                /**
                 * @brief Construct a new Adaptive Moment Estimation Optimizer
                 *
                 * @param learning_rate Learning rate (default: 0.01)
                 * @param beta_1 Exponential decay rate for first moment (default: 0.9)
                 * @param beta_2 Exponential decay rate for second moment (default: 0.999)
                 * @param epsilon Small value for numerical stability (default: 1e-8)
                 */
                AdaptiveMomentEstimation(const double& _learning_rate = 0.01, 
                                       const double& _beta_1 = 0.9, 
                                       const double& _beta_2 = 0.999, 
                                       const double& _epsilon = 1e-8) 
                    : Base(_learning_rate, _beta_1, _beta_2, _epsilon) 
                {}
                
                /**
                 * @brief Compute the parameter update using Adam algorithm
                 *
                 * @param gradient The gradient for the current parameter
                 * @param index The index of the parameter being updated
                 * @return af::array The update to apply to the parameter
                 */
                af::array GetDelta(const af::array& _gradient,const uint32_t& _index) override {
                    
                    this->m_iteration++;
    
                    this->m_stack_previous_gradient[_index] = 
                        this->m_beta_1 * this->m_stack_previous_gradient[_index] + 
                        (1.0 - this->m_beta_1) * _gradient;
    
                    this->m_vector_velocity[_index] = 
                        this->m_beta_2 * this->m_vector_velocity[_index] + 
                        (1.0 - this->m_beta_2) * af::pow(_gradient, 2);
    
                    this->m_mt = this->m_stack_previous_gradient[_index] / 
                        (1.0 - std::pow(this->m_beta_1, this->m_iteration));
                    
                    this->m_vt = this->m_vector_velocity[_index] / 
                        (1.0 - std::pow(this->m_beta_2, this->m_iteration));
    
                    this->m_stack_previous_gradient[_index].eval();
                    this->m_vector_velocity[_index].eval();
                    this->m_mt.eval();
                    this->m_vt.eval();
    
                    return (this->m_learning_rate * this->m_mt) / (af::sqrt(this->m_vt) + this->m_epsilon);
                }
            };
        }
    }
}