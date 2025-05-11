#pragma once
#include "Base.h"

namespace lantern
{
    namespace optimizer
    {

        class RootMeanSquarePropagation : public Base
        {
        public:
            /**
             * @brief Construct a new Root Mean Square Propagation Optimizer
             *
             * @param learning_rate
             * @param beta
             * @param epsilon
             */
            RootMeanSquarePropagation(double learning_rate = 0.01, double beta_1 = 0.9, double epsilon = 1e-8) : Base(learning_rate, beta_1, 0.999, epsilon) {}

            /**
             * @brief Get the Optimize result of gradient
             *
             * @param gradient
             * @param index
             * @return af::array
             */
            af::array GetDelta(af::array &gradient, uint32_t &index) override
            {
                this->vector_velocity[index] *= this->beta_1;
                this->vector_velocity[index] += this->learning_rate * af::pow(gradient, 2);
                this->vector_velocity[index].eval();
                return this->learning_rate * (gradient / (af::sqrt(this->vector_velocity[index]) + this->epsilon));
            }
        };

    }
}