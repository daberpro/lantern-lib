#pragma once
#include "../pch.h"

namespace lantern {

    namespace init {

        void XavierUnifInit(
            uint32_t& input_size, 
            uint32_t& output_size, 
            af::array& parameters
        ){

            double limit = sqrt(
                6.0f/static_cast<double>(input_size + output_size)
            );

            af::array max = af::max(parameters);
            af::array min = af::min(parameters);
            parameters = ((parameters - min)/(max - min)) * (2.0f * limit) - 1.0f;

        }

        void XavierNormInit(
            uint32_t& input_size, 
            uint32_t& output_size, 
            af::array& parameters
        ){

            double limit = sqrt(
                2.0f/static_cast<double>(input_size + output_size)
            );

            af::array max = af::max(parameters);
            af::array min = af::min(parameters);
            parameters = ((parameters - min)/(max - min)) * (2.0f * limit) - 1.0f;

        }

    }

}