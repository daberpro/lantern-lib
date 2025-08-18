#pragma once
#include <cmath>

/**
 * @defgroup LanternBasicSymbolDerivative Basic implement of derivative function
 */

namespace lantern {

    namespace math {

        /**
         * @brief Derivative of sigmoid
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dsigmoid(const double& x){
            return x * (1-x);
        }

        /**
         * @brief Derivative of natural log
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dlog(const double& x){
            return 1/x;
        }

        /**
         * @brief Derivative of exp
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dexp(const double& x){
            return exp(x);
        }

        /**
         * @brief Derivative of sin
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dsin(const double& x){
            return cos(x);
        }

        /**
         * @brief Derivative of cos
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dcos(const double& x){
            return -sin(x);
        }

        /**
         * @brief Derivative of tan
         * @param x 
         * @return double;
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dtan(const double& x){
            return 1/pow(cos(x),2);
        }

        /**
         * @brief Derivative of ReLU
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double drelu(const double& x){
            return (x > 0? 1 : 0);
        }

        /**
         * @brief Derivative of Swish 
         * @param x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dswish(const double& x){
            return (1 / (1+exp(-x))) + x * dsigmoid(x);
        }

    }

}