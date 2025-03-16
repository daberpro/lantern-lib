#pragma once
#include "../../Perceptron.h"

namespace lantern
{
    namespace perceptron
    {

        namespace optimizer
        {

            class AdaptiveGradientDescent
            {
            private:
                double learning_rate = 0.01, epsilon = 1e-08;
                
            public:
                
                #ifdef OPTIMIZE_VERSION
                /**
                 * @brief Construct a new Adaptive Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 * @param epsilon 
                 */
                AdaptiveGradientDescent(double learning_rate = 0.01, double epsilon = 1e-08) : learning_rate(learning_rate), epsilon(epsilon) {}
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param node 
                 * @param child_index 
                 * @return double 
                 */
                double GetDelta(const double &gradient,lantern::perceptron::Perceptron* node,const uint32_t& child_index)
                {
                    node->stack_prev_gradient[child_index] += pow(node->gradient[child_index],2);
                    return (this->learning_rate / (sqrt(node->stack_prev_gradient[child_index])+this->epsilon)) * gradient;
                }
                #endif

                #ifdef MATRIX_OPTIMIZE
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
                /**
                 * @brief Construct a new Adaptive Gradient Descent Optimizer
                 * 
                 * @param learning_rate 
                 * @param epsilon 
                 */
                AdaptiveGradientDescent(double learning_rate = 0.01, double epsilon = 1e-08) : learning_rate(learning_rate), epsilon(epsilon) {}
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDelta(af::array& gradient, int32_t& index){
                    this->stack_previous_gradient[index] += af::pow(gradient,2);
                    this->stack_previous_gradient[index].eval();
                    return (this->learning_rate / (af::sqrt(this->stack_previous_gradient[index])+this->epsilon)) * gradient;
                }
                #endif

                
            };

        }
    }
}