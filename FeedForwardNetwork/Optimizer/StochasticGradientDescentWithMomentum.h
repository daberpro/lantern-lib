#pragma once
#include "Base.h"

namespace lantern
{
    namespace optimizer
    {

        class StochasticGradientDescentWithMomentum : public Base
        {
        public: 
            /**
             * @brief Construct a new Stochastic Gradient Descent With Momentum Optimizer
             *
             * @param learning_rate
             * @param beta
             */
            StochasticGradientDescentWithMomentum(double learning_rate = 0.01, double beta_1 = 0.9) : Base(learning_rate,beta_1) {}
            
            /**
             * @brief Get the Optimize result of gradient
             *
             * @param gradient
             * @param index
             * @return af::array
             */
            af::array GetDelta(af::array &gradient, uint32_t &index)
            {
                this->vector_velocity[index] *= this->beta_1;
                this->vector_velocity[index] += this->learning_rate * gradient;
                this->vector_velocity[index].eval();
                return this->vector_velocity[index];
            }
        };

    }
}