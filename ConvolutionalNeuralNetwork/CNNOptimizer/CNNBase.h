#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {
    
            class Base {
            protected:
                double m_learning_rate = 0.01, m_beta_1 = 0.9, m_beta_2 = 0.999, m_epsilon = 1e-8;
                uint32_t m_iteration = 0;
                af::array m_w_mt, m_w_vt; // for weigths
                af::array m_b_mt, m_b_vt; // for bias
                af::array m_batch_norm_mt, m_batch_norm_vt; // for batch norm parameters
                lantern::utility::Vector<af::array> m_w_vector_velocity; // for weights
                lantern::utility::Vector<af::array> m_w_stack_previous_gradient; // for weights
                lantern::utility::Vector<af::array> m_b_vector_velocity; // for bias
                lantern::utility::Vector<af::array> m_b_stack_previous_gradient; // for bias
                lantern::utility::Vector<af::array> m_batch_norm_vector_velocity; // for batch norm parameters
                lantern::utility::Vector<af::array> m_batch_norm_stack_previous_gradient; // for batch norm parameters

            public:
    
                Base(const double& _learning_rate = 0.01,const double& _beta_1 = 0.9,const double& _beta_2 = 0.999,const double& _epsilon = 1e-8) : 
                m_learning_rate(_learning_rate), 
                m_beta_1(_beta_1), 
                m_beta_2(_beta_2), 
                m_epsilon(_epsilon) {}
    
                virtual ~Base(){};

                /**
                 * @brief Get learning rate for weights and bias
                 * @return double
                 */
                virtual std::pair<af::array,af::array> GetDelta(const af::array& _gradient_w, const af::array& _gradient_b,const uint32_t& _index){
                    return {af::array(),af::array()};
                };

                /**
                 * @brief Get learning rate for batch norm parameters
                 * @return af::array
                 */
                virtual af::array GetDeltaBatchNorm(const af::array& _batch_norm_gradient,const uint32_t& _index){
                    return af::array();
                };
                
                /**
                 * @brief Get learning rate for weights
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetWVectorVelocity(){
                    return this->m_w_vector_velocity;
                }
                
                /**
                 * @brief Get stack previous gradient for weights
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetWStackPrevGrad(){
                    return this->m_w_stack_previous_gradient;
                }

                /**
                 * @brief Get learning rate for bias
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBVectorVelocity(){
                    return this->m_b_vector_velocity;
                }
                
                /**
                 * @brief Get stack previous gradient for bias
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBStackPrevGrad(){
                    return this->m_b_stack_previous_gradient;
                }

                /**
                 * @brief Get learning rate for other parameters
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBatchNormVectorVelocity(){
                    return this->m_batch_norm_vector_velocity;
                }

                /**
                 * @brief Get stack previous gradient for other parameters
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBatchNormStackPreviousGradient(){
                    return this->m_batch_norm_stack_previous_gradient;
                }
            };
    
        }
    }

}