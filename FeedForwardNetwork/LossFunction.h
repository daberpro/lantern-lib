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
             * @brief Derivative of Sum Squaref Residual
             * 
             * @param output 
             * @param target 
             * @return af::array 
             */
            af::array DerivativeSumSquaredResidual(af::array& output, af::array& target){
                return -2 * (target - output);
            };

        }
    }
}