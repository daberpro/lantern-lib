#pragma once
#include <cmath>

namespace latern {

    namespace math {

        inline double dsigmoid(const double& x){
            return x * (1-x);
        }

        inline double dlog(const double& x){
            return 1/x;
        }

        inline double dexp(const double& x){
            return exp(x);
        }

        inline double dsin(const double& x){
            return cos(x);
        }

        inline double dcos(const double& x){
            return -sin(x);
        }

        inline double dtan(const double& x){
            return 1/pow(cos(x),2);
        }

        inline double drelu(const double& x){
            return (x > 0? 1 : 0);
        }

        inline double dswish(const double& x){
            return (1 / (1+exp(-x))) + x * dsigmoid(x);
        }

    }

}