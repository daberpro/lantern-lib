#pragma once
#include "Perceptron.h"
// already define in Perceptron.h
// #include "../Headers/Vector.h"
#include <Optimizer/GradientDescent.h>
#include <Optimizer/AdaptiveMomentEstimation.h>
#include <Optimizer/RootMeanSquaredPropagation.h>
#include <Optimizer/StochasticGradientDescentWithMomentum.h>
#include <Optimizer/AdaptiveGradientDescent.h>
#include "../Headers/SymbolDerivative.h"

namespace lantern {
    namespace perceptron {

        /**
         * @brief Backpropagation for model
         * 
         * @tparam Optimizer 
         * @param parameters 
         * @param gradient_based_parameters 
         * @param operators 
         * @param outputs 
         * @param opt 
         */
        template <typename Optimizer>
        void BackPropagation(
            lantern::utility::Vector<af::array>& parameters,
            lantern::utility::Vector<af::array>& gradient_based_parameters,
            lantern::utility::Vector<lantern::perceptron::Activation>& operators,
            lantern::utility::Vector<af::array>& outputs,
            Optimizer& opt
        ){

            af::array gradient,gradient_weight, gradient_bias, all_gradient, softmax_matrix_m, softmax_matrix_i;

            for(int32_t i = parameters.size() - 1; i > 0; i--){
                
                af::array& parameter = parameters[i];
                af::array& output = outputs[i], prev_output = outputs[i-1];
                af::array& gradient_based_parameter = gradient_based_parameters[i + 1];
                lantern::perceptron::Activation& op = operators[i - 1];

                switch (op)
                {
                case Activation::SIGMOID:

                    // get gradient from current input
                    gradient = output * ( 1 - output );
                    gradient.eval();
                    break;

                case Activation::RELU:

                    // get gradient from current input
                    gradient = output;
                    gradient(af::where(gradient > 0)) = 1; 
                    break;

                case Activation::SWISH:
                    
                    // get gradient from current input
                    gradient = output +  (output * ( 1 - output )) * output;
                    gradient.eval();
                    break;
                case Activation::SOFTMAX:
                    
                    
                    
                    break;

                    break;
                case Activation::LINEAR:
                    
                    gradient = (output / output);
                    gradient.eval();
                    break;
                }

                if(op == Activation::SOFTMAX){
                    
                    gradient_based_parameters[i] = gradient_based_parameter;

                }else{

                    // multiply gradient with previous layer gradient
                    gradient = gradient * gradient_based_parameter;
                    // matrix multiplication with previous output
                    // with gradient because the pattern of formula
                    // in chain rule
                    gradient_weight = af::matmul(prev_output,gradient.T());
                    // because the derivative of bias is only 1
                    // then just set gradient of bias to be gradient
                    gradient_bias = gradient;
                    // join weight and bias to process with current
                    // parameter
                    all_gradient = af::join(0,gradient_weight,gradient_bias.T());
                    // update parameter using optimizer
                    parameter -= opt.GetDelta(all_gradient,i);
                    parameter.eval();
    
                    // pass throught the current gradient to next layer
                    gradient_based_parameters[i] = af::matmul(
                        parameter(af::seq(0,parameter.dims(0) - 2),af::span),
                        gradient
                    );
                }


            }

        }

        /**
         * @brief Calculate gradient for each layer and save it into batch_gradient
         * 
         * @tparam Optimizer 
         * @param parameters 
         * @param gradient_based_parameters 
         * @param operators 
         * @param outputs 
         * @param opt 
         * @param batch_gradient 
         */
        template <typename Optimizer>
        void CalculateGradient(
            lantern::utility::Vector<af::array>& parameters,
            lantern::utility::Vector<af::array>& gradient_based_parameters,
            lantern::utility::Vector<lantern::perceptron::Activation>& operators,
            lantern::utility::Vector<af::array>& outputs,
            Optimizer& opt,
            lantern::utility::Vector<af::array>& batch_gradient
        ){

            af::array gradient,gradient_weight, gradient_bias, all_gradient, softmax_matrix_m, softmax_matrix_i;

            for(int32_t i = parameters.size() - 1; i > 0; i--){
                
                af::array& parameter = parameters[i];
                af::array& output = outputs[i], prev_output = outputs[i-1];
                af::array& gradient_based_parameter = gradient_based_parameters[i + 1];
                lantern::perceptron::Activation& op = operators[i - 1];

                switch (op)
                {
                case Activation::SIGMOID:

                    // get gradient from current input
                    gradient = output * ( 1 - output );
                    gradient.eval();
                    break;

                case Activation::RELU:

                    // get gradient from current input
                    gradient = output;
                    gradient(af::where(gradient > 0)) = 1; 
                    break;

                case Activation::SWISH:
                    
                    // get gradient from current input
                    gradient = output +  (output * ( 1 - output )) * output;
                    gradient.eval();
                    break;

                case Activation::LINEAR:
                    
                    gradient = (output / output);
                    gradient.eval();
                    break;

                }

                if(op == Activation::SOFTMAX){

                    // pass throught the current gradient to next layer
                    gradient_based_parameters[i] = gradient_based_parameter;
                    
                }else{
                    
                    // multiply gradient with previous layer gradient
                    gradient = gradient * gradient_based_parameter;

                    // matrix multiplication with previous output
                    // with gradient because the pattern of formula
                    // in chain rule
    
                    gradient_weight = af::matmul(prev_output,gradient.T());
    
                    // because the derivative of bias is only 1
                    // then just set gradient of bias to be gradient
                    gradient_bias = gradient;
                    // join weight and bias to process with current
                    // parameter
                    all_gradient = af::join(0,gradient_weight,gradient_bias.T());
                    batch_gradient[i] += all_gradient;
                    batch_gradient[i].eval();
                    
                    // ! DO NOT update parameter using optimizer
                    // parameter -= opt.GetDelta(all_gradient,i);
                    // parameter.eval();
    
                    // pass throught the current gradient to next layer
                    gradient_based_parameters[i] = af::matmul(
                        parameter(af::seq(0,parameter.dims(0) - 2),af::span),
                        gradient
                    );
                }


            }

        }
        
    }
}

