#pragma once
#include "FFNBase.h"

namespace lantern {
    namespace ffn {

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
                af::array GetDelta(const af::array& _gradient, const uint32_t& _index) override {
                    return this->m_learning_rate * _gradient;
                }
            };
        }
    }
}