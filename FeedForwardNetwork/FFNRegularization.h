#pragma once
#include "../pch.h"

namespace lantern {

    namespace regularization {

        /**
         * @brief L2 Regularization (Ideal for neural network)
         * 
         * @param lambda 
         * @param batch_size 
         * @param weights this is not derivative of weights but actual weights
         */
        af::array L2Regularization(
            const double& lambda, 
            const double& batch_size,
            const af::array& weights
        ){

            /**
             * actual function
             * (lambda/ (2* batch_size)) * weights ^ 2
             */
            return (lambda/batch_size) * weights;

        }

        /**
         * @brief L1 Regularization (Ideal for neural network)
         * 
         * @param lambda 
         * @param batch_size 
         * @param weights this is not derivative of weights but actual weights
         */
        af::array L1Regularization(
            const double& lambda, 
            const double& batch_size,
            const af::array& weights
        ){

            /**
             * actual function
             * (lambda/ batch_size) * abs(weights)
             * remember abs(x) function when x < 0 the output will be -x
             * and if x > 0 the output will be x 
             * and if x = 0 the output will be 0, or sign(x)
             */
            return (lambda/batch_size) * af::sign(weights);

        }

        


    }

}