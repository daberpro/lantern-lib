#pragma once

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
                GradientDescent(double learning_rate = 0.01f): Base(learning_rate) {}
                
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDelta(af::array& gradient, uint32_t& index) override {
                    return this->learning_rate * gradient;
                }
            };

        }
    }
}