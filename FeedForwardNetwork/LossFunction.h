#pragma once
#include "../pch.h"

namespace lantern {

    namespace perceptron {
        
        namespace loss {

            #ifdef MATRIX_OPTIMIZE
            double SumSquaredResidual(af::array& output, af::array& target){
                return af::sum(af::pow(target - output,2)).scalar<double>();
            }

            af::array DerivativeSumSquaredResidual(af::array& output, af::array& target){
                return -2 * (target - output);
            };
            #endif

        }
    }
}