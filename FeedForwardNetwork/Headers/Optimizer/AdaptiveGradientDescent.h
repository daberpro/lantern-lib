#pragma once
#include "../../Perceptron.h"

namespace latern
{
    namespace perceptron
    {

        namespace optimizer
        {

            class AdaptiveGradientDescent
            {
            private:
                double learning_rate = 0.01, epsilon = 1e-08;
                uint32_t iteration = 0;

            public:
                AdaptiveGradientDescent(double learning_rate = 0.01, double epsilon = 1e-08) : learning_rate(learning_rate), epsilon(epsilon) {}
                double GetDelta(const double &gradient,latern::perceptron::Perceptron* node,const uint32_t& child_index)
                {
                    node->stack_prev_gradient(child_index, 0) += pow(node->gradient(child_index, 0),2);
                    return (this->learning_rate / (sqrt(node->stack_prev_gradient(child_index, 0).scalar<double>())+this->epsilon)) * gradient;
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