#pragma once
#include "../../Perceptron.h"

namespace latern
{
    namespace perceptron
    {

        namespace optimizer
        {

            class RootMeanSquarePropagation
            {
            private:
                double learning_rate = 0.01, beta = 0.9, epsilon = 1e-8;
                uint32_t iteration = 0;

            public:
                RootMeanSquarePropagation(double learning_rate = 0.01, double beta = 0.9,double epsilon = 1e-8) : learning_rate(learning_rate), beta(beta), epsilon(epsilon) {}
                double GetDelta(const double &gradient,latern::perceptron::Perceptron* node,const uint32_t& child_index)
                {
                    node->vector_velocity[child_index] = node->vector_velocity[child_index] * this->beta + this->learning_rate * pow(gradient,2);
                    return this->learning_rate * (gradient/(sqrt(node->vector_velocity[child_index])+this->epsilon));
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