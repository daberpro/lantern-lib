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
                AdaptiveMomentEstimation(double learning_rate = 0.01, 
                                       double beta_1 = 0.9, 
                                       double beta_2 = 0.999, 
                                       double epsilon = 1e-8) 
                    : Base(learning_rate, beta_1, beta_2, epsilon) 
                {}
                
                /**
                 * @brief Compute the parameter update using Adam algorithm
                 *
                 * @param gradient The gradient for the current parameter
                 * @param index The index of the parameter being updated
                 * @return af::array The update to apply to the parameter
                 */
                af::array GetDelta(af::array& gradient, uint32_t& index) override {
                    
                    iteration++;
    
                    stack_previous_gradient[index] = 
                        this->beta_1 * this->stack_previous_gradient[index] + 
                        (1.0 - this->beta_1) * gradient;
    
                    this->vector_velocity[index] = 
                        beta_2 * this->vector_velocity[index] + 
                        (1.0 - this->beta_2) * af::pow(gradient, 2);
    
                    this->mt = this->stack_previous_gradient[index] / 
                        (1.0 - std::pow(this->beta_1, this->iteration));
                    
                    this->vt = vector_velocity[index] / 
                        (1.0 - std::pow(this->beta_2, this->iteration));
    
                    stack_previous_gradient[index].eval();
                    vector_velocity[index].eval();
                    mt.eval();
                    vt.eval();
    
                    return (this->learning_rate * this->mt) / (af::sqrt(this->vt) + this->epsilon);
                }
            };
        }
    }
}