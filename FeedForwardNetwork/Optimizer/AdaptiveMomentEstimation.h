#pragma once
#include "Base.h"

namespace lantern
{
    namespace optimizer
    {

        class AdaptiveMomentEstimation : public Base
        {
        public:
            
            /**
             * @brief Construct a new Adaptive Moment Estimation Optimizer
             *
             * @param learning_rate
             * @param beta_1
             * @param beta_2
             * @param epsilon
             */
            AdaptiveMomentEstimation(double learning_rate = 0.01, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8) : Base(learning_rate, beta_1, beta_2, epsilon) {}
            
            /**
             * @brief Get the Optimize result of gradient
             *
             * @param gradient
             * @param index
             * @return af::array
             */
            af::array GetDelta(af::array &gradient, uint32_t &index) override
            {
                this->iteration++;
                this->stack_previous_gradient[index] = this->stack_previous_gradient[index] * this->beta_1 + (1.0 - this->beta_1) * gradient;
                this->stack_previous_gradient[index].eval();
                this->vector_velocity[index] = this->vector_velocity[index] * this->beta_2 + (1.0 - this->beta_2) * pow(gradient, 2);
                this->vector_velocity[index].eval();

                mt = this->stack_previous_gradient[index] / (1.0 - pow(this->beta_1, this->iteration));
                vt = this->vector_velocity[index] / (1.0 - pow(this->beta_2, this->iteration));
                mt.eval();
                vt.eval();
                return (this->learning_rate * mt) / (sqrt(vt) + this->epsilon);
            }
        };

    }
}