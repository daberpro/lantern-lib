#pragma once
#include "../../Perceptron.h"

namespace lantern
{
    namespace perceptron
    {

        namespace optimizer
        {

            class StochasticGradientDescentWithMomentum
            {
            private:
                double learning_rate = 0.01, beta = 0.9;
                
            public:
                #ifdef OPTIMIZE_VERSION
                StochasticGradientDescentWithMomentum(double learning_rate = 0.01, double beta = 0.9) : learning_rate(learning_rate), beta(beta) {}
                double GetDelta(const double &gradient,lantern::perceptron::Perceptron* node,const uint32_t& child_index)
                {
                    node->vector_velocity[child_index] = node->vector_velocity[child_index] * this->beta + this->learning_rate * gradient;
                    return node->vector_velocity[child_index];
                }
                #endif

                #ifdef MATRIX_OPTIMIZE
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
                StochasticGradientDescentWithMomentum(double learning_rate = 0.01, double beta = 0.9) : learning_rate(learning_rate), beta(beta) {}
                af::array GetDelta(af::array& gradient, int32_t& index)
                {
                    this->vector_velocity[index] *= this->beta;
                    this->vector_velocity[index] += this->learning_rate * gradient;
                    this->vector_velocity[index].eval();
                    return this->vector_velocity[index];
                }
                #endif
            };

        }
    }
}