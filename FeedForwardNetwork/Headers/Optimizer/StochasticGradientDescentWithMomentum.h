#pragma once
#include "../../Perceptron.h"

namespace latern
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
                StochasticGradientDescentWithMomentum(double learning_rate = 0.01, double beta = 0.9) : learning_rate(learning_rate), beta(beta) {}
                double GetDelta(const double &gradient,latern::perceptron::Perceptron* node,const uint32_t& child_index)
                {
                    node->vector_velocity(child_index, 0) = node->vector_velocity(child_index, 0) * this->beta + this->learning_rate * gradient;
                    return node->vector_velocity(child_index, 0).scalar<double>();
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