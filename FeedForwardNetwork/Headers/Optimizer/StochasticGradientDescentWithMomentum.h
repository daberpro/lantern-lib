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
                uint32_t iteration = 0;

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
                StochasticGradientDescentWithMomentum(double learning_rate = 0.01, double beta = 0.9) : learning_rate(learning_rate), beta(beta) {}
                double GetDelta(af::array& gradient,af::array& vector_velocity)
                {
                    vector_velocity = vector_velocity * this->beta + this->learning_rate * gradient;
                    return vector_velocity;
                }
                #endif

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