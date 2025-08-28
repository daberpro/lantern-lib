#pragma once
#include "../pch.h"

/**
 * @defgroup LantermFFNRegularize Regularization for lantern FFN
 */

namespace lantern {

    namespace regularization {

        /**
         * @brief L2 Regularization (Ideal for neural network)
         * 
         * @param lambda 
         * @param batch_size 
         * @param weights this is not derivative of weights but actual weights
         * @ingroup LantermFFNRegularize
         */
        inline af::array L2Regularization(
            const double& _lambda, 
            const double& _batch_size,
            const af::array& _weights
        ){

            /**
             * actual function
             * (lambda/ (2* batch_size)) * weights ^ 2
             */
            return (_lambda/_batch_size) * _weights;

        }

        /**
         * @brief L1 Regularization (Ideal for neural network)
         * 
         * @param lambda 
         * @param batch_size 
         * @param weights this is not derivative of weights but actual weights
         * @ingroup LantermFFNRegularize
         */
        inline af::array L1Regularization(
            const double& _lambda, 
            const double& _batch_size,
            const af::array& _weights
        ){

            /**
             * actual function
             * (lambda/ batch_size) * abs(weights)
             * remember abs(x) function when x < 0 the output will be -x
             * and if x > 0 the output will be x 
             * and if x = 0 the output will be 0, or sign(x)
             */
            return (_lambda/_batch_size) * af::sign(_weights);

        }

        


    }

}