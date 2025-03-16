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
                double learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8;
                uint32_t iteration = 0;
            public:
                #ifdef OPTIMIZE_VERSION
                double mt = 0, vt = 0;
                /**
                 * @brief Construct a new Adaptive Moment Estimation Optimizer
                 * 
                 * @param learning_rate 
                 * @param beta_1 
                 * @param beta_2 
                 * @param epsilon 
                 */
                AdaptiveMomentEstimation(double learning_rate = 0.01, double beta_1 = 0.9, double beta_2 = 0.999,double epsilon = 1e-8) : learning_rate(learning_rate), beta_1(beta_1), beta_2(beta_2), epsilon(epsilon) {}
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
                    this->iteration++;
                    node->stack_prev_gradient[child_index] = node->stack_prev_gradient[child_index] * this->beta_1 + (1.0 - this->beta_1) * gradient;
                    node->vector_velocity[child_index] = node->vector_velocity[child_index] * this->beta_2 + (1.0 - this->beta_2) * pow(gradient,2);
                    mt = node->stack_prev_gradient[child_index]/(1.0 - pow(this->beta_1,this->iteration));
                    vt = node->vector_velocity[child_index]/(1.0 - pow(this->beta_2,this->iteration));
                    return (this->learning_rate * mt)/(sqrt(vt) + this->epsilon);
                }
                #endif

                #ifdef MATRIX_OPTIMIZE
                af::array mt, vt;
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
                /**
                 * @brief Construct a new Adaptive Moment Estimation Optimizer
                 * 
                 * @param learning_rate 
                 * @param beta_1 
                 * @param beta_2 
                 * @param epsilon 
                 */
                AdaptiveMomentEstimation(double learning_rate = 0.01, double beta_1 = 0.9, double beta_2 = 0.999,double epsilon = 1e-8) : learning_rate(learning_rate), beta_1(beta_1), beta_2(beta_2), epsilon(epsilon) {}
                /**
                 * @brief Get the Optimize result of gradient
                 * 
                 * @param gradient 
                 * @param index 
                 * @return af::array 
                 */
                af::array GetDelta(af::array& gradient, int32_t& index)
                {
                    this->iteration++;
                    this->stack_previous_gradient[index] = this->stack_previous_gradient[index] * this->beta_1 + (1.0 - this->beta_1) * gradient;
                    this->stack_previous_gradient[index].eval();
                    this->vector_velocity[index] = this->vector_velocity[index] * this->beta_2 + (1.0 - this->beta_2) * pow(gradient,2);
                    this->vector_velocity[index].eval();

                    mt = this->stack_previous_gradient[index] / (1.0 - pow(this->beta_1,this->iteration));
                    vt = this->vector_velocity[index] / (1.0 - pow(this->beta_2,this->iteration));
                    mt.eval();
                    vt.eval();
                    return (this->learning_rate * mt)/(sqrt(vt) + this->epsilon);
                }
                #endif

            };

        }
    }
}