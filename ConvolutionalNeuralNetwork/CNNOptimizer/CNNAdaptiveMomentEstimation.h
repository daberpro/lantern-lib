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
                AdaptiveMomentEstimation(const double& learning_rate = 0.01f): Base(learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& gradient_w, const af::array& gradient_b,const uint32_t& index) override {
                    this->iteration++;
    
                    this->w_stack_previous_gradient[index] = 
                        this->beta_1 * this->w_stack_previous_gradient[index] + 
                        (1.0 - this->beta_1) * gradient_w;
    
                    this->w_vector_velocity[index] = 
                        beta_2 * this->w_vector_velocity[index] + 
                        (1.0 - this->beta_2) * af::pow(gradient_w, 2);
    
                    this->w_mt = this->w_stack_previous_gradient[index] / 
                        (1.0 - std::pow(this->beta_1, this->iteration));
                    
                    this->w_vt = w_vector_velocity[index] / 
                        (1.0 - std::pow(this->beta_2, this->iteration));
    
                    this->w_stack_previous_gradient[index].eval();
                    this->w_vector_velocity[index].eval();
                    this->w_mt.eval();
                    this->w_vt.eval();

                    this->b_stack_previous_gradient[index] = 
                        this->beta_1 * this->b_stack_previous_gradient[index] + 
                        (1.0 - this->beta_1) * gradient_b;
    
                    this->b_vector_velocity[index] = 
                        beta_2 * this->b_vector_velocity[index] + 
                        (1.0 - this->beta_2) * af::pow(gradient_b, 2);
    
                    this->b_mt = this->b_stack_previous_gradient[index] / 
                        (1.0 - std::pow(this->beta_1, this->iteration));
                    
                    this->b_vt = b_vector_velocity[index] / 
                        (1.0 - std::pow(this->beta_2, this->iteration));
    
                    this->b_stack_previous_gradient[index].eval();
                    this->b_vector_velocity[index].eval();
                    this->b_mt.eval();
                    this->b_vt.eval();
    
                    return {
                        (this->learning_rate * this->w_mt) / (af::sqrt(this->w_vt) + this->epsilon),
                        (this->learning_rate * this->b_mt) / (af::sqrt(this->b_vt) + this->epsilon)
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
                    this->batch_norm_stack_previous_gradient[index] = 
                        this->beta_1 * this->batch_norm_stack_previous_gradient[index] + 
                        (1.0 - this->beta_1) * batch_norm_gradient;
    
                    this->batch_norm_vector_velocity[index] = 
                        beta_2 * this->batch_norm_vector_velocity[index] + 
                        (1.0 - this->beta_2) * af::pow(batch_norm_gradient, 2);
    
                    this->batch_norm_mt = this->batch_norm_stack_previous_gradient[index] / 
                        (1.0 - std::pow(this->beta_1, this->iteration));
                    
                    this->batch_norm_vt = batch_norm_vector_velocity[index] / 
                        (1.0 - std::pow(this->beta_2, this->iteration));
    
                    this->batch_norm_stack_previous_gradient[index].eval();
                    this->batch_norm_vector_velocity[index].eval();
                    this->batch_norm_mt.eval();
                    this->batch_norm_vt.eval();
                    
                    return (this->learning_rate * this->batch_norm_mt) / (af::sqrt(this->batch_norm_vt) + this->epsilon);
                }
            };

        }
    }
}