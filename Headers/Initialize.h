#pragma once
#include "../pch.h"

/**
 * @defgroup LanternInitFunction Initalize function for parameters using Xavier/Glorot method
 */

namespace lantern {

    namespace init {

        /**
         * @brief Xavier Uniform distribution intialize
         * @param input_size 
         * @param output_size 
         * @param parameters 
         * @ingroup LanternInitFunction
         */
        inline void XavierUnifInit(
            const uint32_t& input_size, 
            const uint32_t& output_size, 
            af::array& parameters
        ) {
            double limit = sqrt(6.0 / (input_size + output_size));
            parameters = af::randu(parameters.dims(), f64) * (2.0 * limit) - limit;
        }

        /**
         * @brief Xavier Normal distribution initialize
         * @param input_size 
         * @param output_size 
         * @param parameters 
         * @ingroup LanternInitFunction
         */
        inline void XavierNormInit(
            const uint32_t& input_size, 
            const uint32_t& output_size, 
            af::array& parameters
        ) {
            double stddev = sqrt(2.0 / (input_size + output_size));
            parameters = af::randn(parameters.dims(), f64) * stddev;
        }

    }

}
