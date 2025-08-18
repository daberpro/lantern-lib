#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {

    namespace cnn {

        namespace optimizer {
    
            class Base {
            protected:
                double learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8;
                uint32_t iteration = 0;
                af::array w_mt, w_vt; // for weigths
                af::array b_mt, b_vt; // for bias
                af::array batch_norm_mt, batch_norm_vt; // for batch norm parameters
                lantern::utility::Vector<af::array> w_vector_velocity; // for weights
                lantern::utility::Vector<af::array> w_stack_previous_gradient; // for weights
                lantern::utility::Vector<af::array> b_vector_velocity; // for bias
                lantern::utility::Vector<af::array> b_stack_previous_gradient; // for bias
                lantern::utility::Vector<af::array> batch_norm_vector_velocity; // for batch norm parameters
                lantern::utility::Vector<af::array> batch_norm_stack_previous_gradient; // for batch norm parameters

            public:
    
                Base(const double& learning_rate = 0.01,const double& beta_1 = 0.9,const double& beta_2 = 0.999,const double& epsilon = 1e-8) : 
                learning_rate(learning_rate), 
                beta_1(beta_1), 
                beta_2(beta_2), 
                epsilon(epsilon) {}
    
                virtual ~Base(){};

                /**
                 * @brief Get learning rate for weights and bias
                 * @return double
                 */
                virtual std::pair<af::array,af::array> GetDelta(const af::array& gradient_w, const af::array& gradient_b,const uint32_t& index){
                    return {af::array(),af::array()};
                };

                /**
                 * @brief Get learning rate for batch norm parameters
                 * @return af::array
                 */
                virtual af::array GetDeltaBatchNorm(const af::array& batch_norm_gradient,const uint32_t& index){
                    return af::array();
                };
                
                /**
                 * @brief Get learning rate for weights
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetWVectorVelocity(){
                    return this->w_vector_velocity;
                }
                
                /**
                 * @brief Get stack previous gradient for weights
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetWStackPrevGrad(){
                    return this->w_stack_previous_gradient;
                }

                /**
                 * @brief Get learning rate for bias
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBVectorVelocity(){
                    return this->b_vector_velocity;
                }
                
                /**
                 * @brief Get stack previous gradient for bias
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBStackPrevGrad(){
                    return this->b_stack_previous_gradient;
                }

                /**
                 * @brief Get learning rate for other parameters
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBatchNormVectorVelocity(){
                    return this->batch_norm_vector_velocity;
                }

                /**
                 * @brief Get stack previous gradient for other parameters
                 * @return lantern::utility::Vector<af::array>&
                 */
                lantern::utility::Vector<af::array>& GetBatchNormStackPreviousGradient(){
                    return this->batch_norm_stack_previous_gradient;
                }
            };
    
        }
    }

}