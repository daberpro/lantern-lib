#pragma once
#include "../pch.h"

namespace lantern {

    namespace perceptron {
        
        namespace loss {

            /**
             * @brief Sum Squared Residual Error
             * 
             * @param output 
             * @param target 
             * @return double 
             */
            double SumSquaredResidual(af::array& output, af::array& target){
                return af::sum(af::pow(target - output,2)).scalar<double>();
            }

            /**
             * @brief Derivative of Sum Square Residual
             * 
             * @param output 
             * @param target 
             * @return af::array 
             */
            af::array DerivativeSumSquaredResidual(af::array& output, af::array& target){
                return -2 * (target - output);
            }

            /**
             * @brief Cross Entropy Loss Function
             * 
             * @param output 
             * @param target 
             * @return double 
             */
            double CrossEntropy(af::array& output, af::array& target) {
                // Clip output to avoid log(0)
                af::array clipped_output = af::max(output, 1e-12);  // prevent log(0)
                clipped_output = af::min(clipped_output, 1.0);      // ensure <= 1
                return (-af::sum(target * af::log(clipped_output))).scalar<double>() / output.dims(0);
            }

            /**
             * @brief Derivative Of Cross Entropy Loss Function
             * 
             * @param output 
             * @param target 
             * @return af::array 
             */
            af::array DerivativeCrossEntropy(af::array& output, af::array& target) {
                
                /**
                 * just return the output, because for softmax + crossentropy
                 * gradient already calculate in FeedForwardNetwork.h
                 */

                return output;

            }

        }
    }
}