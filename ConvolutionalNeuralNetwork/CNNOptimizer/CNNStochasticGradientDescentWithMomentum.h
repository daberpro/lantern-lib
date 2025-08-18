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
                StochasticGradientDescentWithMomentum(const double& learning_rate = 0.01f): Base(learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& gradient_w, const af::array& gradient_b,const uint32_t& index) override {
                    //for weights
                    this->w_vector_velocity[index] *= this->beta_1;
                    this->w_vector_velocity[index] += this->learning_rate * gradient_w;
                    this->w_vector_velocity[index].eval();

                    // for bias
                    this->b_vector_velocity[index] *= this->beta_1;
                    this->b_vector_velocity[index] += this->learning_rate * gradient_b;
                    this->b_vector_velocity[index].eval();

                    return {
                        this->w_vector_velocity[index],
                        this->b_vector_velocity[index]
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
                    // for other parameters
                    this->batch_norm_vector_velocity[index] *= this->beta_1;
                    this->batch_norm_vector_velocity[index] += this->learning_rate * batch_norm_gradient;
                    this->batch_norm_vector_velocity[index].eval();
                    
                    return this->batch_norm_vector_velocity[index];
                }
            };

        }
    }
}