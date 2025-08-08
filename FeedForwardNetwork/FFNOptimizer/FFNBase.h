#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {

    namespace ffn {

        namespace optimizer {
    
            class Base {
            protected:
                double learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8;
                uint32_t iteration = 0;
            public:
                af::array mt, vt;
                lantern::utility::Vector<af::array> vector_velocity;
                lantern::utility::Vector<af::array> stack_previous_gradient;
    
                Base(double learning_rate = 0.01, double beta_1 = 0.9, double beta_2 = 0.999,double epsilon = 1e-8) : 
                learning_rate(learning_rate), 
                beta_1(beta_1), 
                beta_2(beta_2), 
                epsilon(epsilon) {}
    
                virtual ~Base(){};
                virtual af::array GetDelta(af::array& gradient, uint32_t& index){
                    return af::array();
                };
    
                lantern::utility::Vector<af::array>& GetVectorVelocity(){
                    return this->vector_velocity;
                }
                
                lantern::utility::Vector<af::array>& GetStackPrevGrad(){
                    return this->stack_previous_gradient;
                }
            };
    
        }
    }

}