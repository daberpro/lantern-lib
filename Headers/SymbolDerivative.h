#pragma once
#include <cmath>

/**
 * @defgroup LanternBasicSymbolDerivative Basic implement of derivative function
 */

namespace lantern {

    namespace math {

        /**
         * @brief Derivative of sigmoid
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dsigmoid(const double& _x){
            return _x * (1-_x);
        }

        /**
         * @brief Derivative of natural log
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dlog(const double& _x){
            return 1/_x;
        }

        /**
         * @brief Derivative of e_xp
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double de_xp(const double& _x){
            return exp(_x);
        }

        /**
         * @brief Derivative of sin
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dsin(const double& _x){
            return cos(_x);
        }

        /**
         * @brief Derivative of cos
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dcos(const double& _x){
            return -sin(_x);
        }

        /**
         * @brief Derivative of tan
         * @param _x 
         * @return double;
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dtan(const double& _x){
            return 1/pow(cos(_x),2);
        }

        /**
         * @brief Derivative of ReLU
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double drelu(const double& _x){
            return (_x > 0? 1 : 0);
        }

        /**
         * @brief Derivative of Swish 
         * @param _x 
         * @return double
         * @ingroup LanternBasicSymbolDerivative
         */
        inline double dswish(const double& _x){
            return (1 / (1+exp(-_x))) + _x * dsigmoid(_x);
        }

    }

}