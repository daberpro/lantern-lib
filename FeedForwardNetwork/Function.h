#pragma once
#include "../pch.h"

namespace lantern{

    namespace activation {

        af::array Sigmoid(af::array& value){
            return 1/(1 + af::exp(-value));
        }

        af::array Linear(af::array& value){
            return value;
        }

        af::array ReLU(af::array& value){
            value(af::where(value <= 0)) = 0;
            return value;
        }

        af::array TanH(af::array& value){
            return (af::exp(value) - af::exp(-value))/(af::exp(value) + af::exp(-value));
        }

        af::array Swish(af::array& value){
            return value * Sigmoid(value);
        }

        af::array SoftMax(af::array& value){
            return af::exp(value)/af::sum(af::exp(value));
        }

    }

    namespace probability {

    }

    namespace loss {

        double SumSquareResidual(af::array& output, af::array& target){
            return af::pow(target - output,2).scalar<double>();
        }

        double CrossEntropy(af::array& output, af::array& target){
            return af::sum(-(target * af::log(output + 1e-012))).scalar<double>();
        }

    }

    namespace derivative {

        af::array Sigmoid(af::array& value){
            return value * (1 - value);
        }

        af::array Swish(af::array& value){
            return activation::Sigmoid(value) +  (value * (1 - value)) * value;
        }

        af::array Linear(af::array& value){
            return af::constant(1.0f,value.dims(0),value.dims(1),f64);
        }

        af::array SumSquareResidual(af::array& output, af::array& target){
            return -2 * (target - output);
        }

    }

}