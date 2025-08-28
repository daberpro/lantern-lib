#pragma once
#include "CNNBase.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {

            class GradientDescent : public Base {
            public:
                /**
                 * @brief Construct a new Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 */
                GradientDescent(const double& _learning_rate = 0.01f): Base(_learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                std::pair<af::array,af::array> GetDelta(const af::array& _gradient_w, const af::array& _gradient_b,const uint32_t& _index) override {
                    return {
                        this->m_learning_rate * _gradient_w,
                        this->m_learning_rate * _gradient_b
                    };
                }

                /**
                 * @brief Get the Optimize result of other parameters
                 * 
                 * @param gradient_other 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDeltaBatchNorm(const af::array& _batch_norm_gradient, const uint32_t& _index) override {
                    return this->m_learning_rate * _batch_norm_gradient;
                }
            };

        }
    }
}