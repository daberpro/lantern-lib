#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {

    namespace ffn {

        namespace optimizer {
    
            class Base {
            protected:
                double m_learning_rate = 0.01, m_beta_1 = 0.9, m_beta_2 = 0.999, m_epsilon = 1e-8;
                uint32_t m_iteration = 0;
                af::array m_mt, m_vt;
                lantern::utility::Vector<af::array> m_vector_velocity;
                lantern::utility::Vector<af::array> m_stack_previous_gradient;

            public:
    
                Base(const double& _learning_rate = 0.01,const double& _beta_1 = 0.9,const double& _beta_2 = 0.999,const double& _epsilon = 1e-8) : 
                m_learning_rate(_learning_rate), 
                m_beta_1(_beta_1), 
                m_beta_2(_beta_2), 
                m_epsilon(_epsilon) {}
    
                virtual ~Base(){};
                virtual af::array GetDelta(const af::array& _gradient, const uint32_t& _index){
                    return af::array();
                };
    
                lantern::utility::Vector<af::array>& GetVectorVelocity(){
                    return this->m_vector_velocity;
                }
                
                lantern::utility::Vector<af::array>& GetStackPrevGrad(){
                    return this->m_stack_previous_gradient;
                }
            };
    
        }
    }

}