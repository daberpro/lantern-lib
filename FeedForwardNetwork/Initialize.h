#pragma once
#include "../pch.h"

namespace lantern {

    namespace init {

        void XavierUnifInit(
            const uint32_t& input_size, 
            const uint32_t& output_size, 
            af::array& parameters
        ) {
            double limit = sqrt(6.0 / (input_size + output_size));
            parameters = af::randu(parameters.dims(), f64) * (2.0 * limit) - limit;
        }

        void XavierNormInit(
            const uint32_t& input_size, 
            const uint32_t& output_size, 
            af::array& parameters
        ) {
            double stddev = sqrt(2.0 / (input_size + output_size));
            parameters = af::randn(parameters.dims(), f64) * stddev;
        }

    }

}
