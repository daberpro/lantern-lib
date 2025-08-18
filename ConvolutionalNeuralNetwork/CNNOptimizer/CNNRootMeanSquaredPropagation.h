#pragma once
#include "CNNBase.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {

            class RootMeanSquaredPropagation : public Base {
            public:
                /**
                 * @brief Construct a new Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 */
                RootMeanSquaredPropagation(const double& learning_rate = 0.01f): Base(learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& gradient_w, const af::array& gradient_b,const uint32_t& index) override {
                    this->w_vector_velocity[index] *= this->beta_1;
                    this->w_vector_velocity[index] += this->learning_rate * af::pow(gradient_w, 2);
                    this->w_vector_velocity[index].eval();

                    this->b_vector_velocity[index] *= this->beta_1;
                    this->b_vector_velocity[index] += this->learning_rate * af::pow(gradient_b, 2);
                    this->b_vector_velocity[index].eval();
                    return {
                        this->learning_rate * (gradient_w / (af::sqrt(this->w_vector_velocity[index]) + this->epsilon)),
                        this->learning_rate * (gradient_b / (af::sqrt(this->b_vector_velocity[index]) + this->epsilon))
                    };
                }

                /**
                 * @brief Get the Optimize result of other parameters
                 * 
                 * @param gradient_other 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDeltaBatchNorm(const af::array& batch_norm_gradient, const uint32_t& index) override {
                    this->batch_norm_vector_velocity[index] *= this->beta_1;
                    this->batch_norm_vector_velocity[index] += this->learning_rate * af::pow(batch_norm_gradient, 2);
                    this->batch_norm_vector_velocity[index].eval();

                    return this->learning_rate * (batch_norm_gradient / (af::sqrt(this->batch_norm_vector_velocity[index]) + this->epsilon));
                }
            };

        }
    }
}