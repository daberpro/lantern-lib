#pragma once
#include "../pch.h"

/**
 * @defgroup LanternInitFunction Initalize function for _parameters using Xavier/Glorot method
 */

namespace lantern {

    namespace init {

        /**
         * @brief Xavier Uniform distribution intialize
         * @param _input_size 
         * @param _output_size 
         * @param _parameters 
         * @ingroup LanternInitFunction
         */
        inline void XavierUnifInit(
            const uint32_t& _input_size, 
            const uint32_t& _output_size, 
            af::array& _parameters
        ) {
            double limit_ = sqrt(6.0 / (_input_size + _output_size));
            _parameters = af::randu(_parameters.dims(), f64) * (2.0 * limit_) - limit_;
        }

        /**
         * @brief Xavier Normal distribution initialize
         * @param _input_size 
         * @param _output_size 
         * @param _parameters 
         * @ingroup LanternInitFunction
         */
        inline void XavierNormInit(
            const uint32_t& _input_size, 
            const uint32_t& _output_size, 
            af::array& _parameters
        ) {
            double stddev_ = sqrt(2.0 / (_input_size + _output_size));
            _parameters = af::randn(_parameters.dims(), f64) * stddev_;
        }

    }

}
