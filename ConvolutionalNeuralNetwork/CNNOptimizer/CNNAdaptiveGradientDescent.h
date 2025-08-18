#pragma once
#include "CNNBase.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {

            class AdaptiveGradientDescent : public Base {
            public:
                /**
                 * @brief Construct a new Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 */
                AdaptiveGradientDescent(const double& learning_rate = 0.01f): Base(learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& gradient_w, const af::array& gradient_b,const uint32_t& index) override {
                    this->w_stack_previous_gradient[index] += af::pow(gradient_w, 2);
                    this->w_stack_previous_gradient[index].eval();

                    this->b_stack_previous_gradient[index] += af::pow(gradient_b, 2);
                    this->b_stack_previous_gradient[index].eval();
                    return {
                        (this->learning_rate / (af::sqrt(this->w_stack_previous_gradient[index]) + this->epsilon)) * gradient_w,
                        (this->learning_rate / (af::sqrt(this->b_stack_previous_gradient[index]) + this->epsilon)) * gradient_b
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
                    this->batch_norm_vector_velocity[index] += af::pow(batch_norm_gradient, 2);
                    this->batch_norm_vector_velocity[index].eval();
                    
                    return (this->learning_rate / (af::sqrt(this->batch_norm_vector_velocity[index]) + this->epsilon)) * batch_norm_gradient;
                }
            };

        }
    }
}