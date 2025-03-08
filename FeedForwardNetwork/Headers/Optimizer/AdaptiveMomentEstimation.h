#pragma once
#include "../../Perceptron.h"

namespace lantern
{
    namespace perceptron
    {

        namespace optimizer
        {

            class AdaptiveMomentEstimation
            {
            private:
                double learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, mt = 0, vt = 0;
                uint32_t iteration = 1;

            public:
                AdaptiveMomentEstimation(double learning_rate = 0.01, double beta_1 = 0.9, double beta_2 = 0.999,double epsilon = 1e-8) : learning_rate(learning_rate), beta_1(beta_1), beta_2(beta_2), epsilon(epsilon) {}
                double GetDelta(const double &gradient,lantern::perceptron::Perceptron* node,const uint32_t& child_index)
                {
                    this->iteration++;
                    node->stack_prev_gradient[child_index] = node->stack_prev_gradient[child_index] * this->beta_1 + (1.0 - this->beta_1) * gradient;
                    node->vector_velocity[child_index] = node->vector_velocity[child_index] * this->beta_2 + (1.0 - this->beta_2) * pow(gradient,2);
                    mt = node->stack_prev_gradient[child_index]/(1.0 - pow(this->beta_1,this->iteration));
                    vt = node->vector_velocity[child_index]/(1.0 - pow(this->beta_2,this->iteration));
                    return (this->learning_rate * mt)/(sqrt(vt) + this->epsilon);
                }

                void SetIteration(const uint32_t& iter){
                    this->iteration = iter;
                }

                uint32_t GetIteration(){
                    return this->iteration;
                }
            };

        }
    }
}