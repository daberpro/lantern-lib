#pragma once
#include "../pch.h"
#include "Perceptron.h"
// already define in Perceptron.h
// #include "../Headers/Vector.h"

namespace lantern
{

    namespace perceptron
    {

        #ifdef OPTIMIZE_VERSION 
        void PerceptronUpdateCalculation(
            Perceptron *node
        ){
            if (node->parents.empty() || node->op == Activation::NOTHING)
            {
                return;
            }

            double value = 0.0;
            Perceptron* parent = nullptr;
            uint32_t i = 0;
            for (; i < node->parents.size(); i++)
            {
                parent = node->parents[i];
                value += parent->value * parent->gradient[node->child_index[i]];    
            }
            /**
             * add bias to the sum product of weight and input
             */
            value += node->gradient[(node->total_gradient_size == 0? 1 : node->total_gradient_size)];
            
            
            switch (node->op)
            {
            case Activation::NATURAL_LOG:
                node->value = log(value);
                break;
            case Activation::EXP:
                node->value = exp(value);
                break;
            case Activation::SIN:
                node->value = sin(value);
                break;
            case Activation::COS:
                node->value = cos(value);
                break;
            case Activation::TAN:
                node->value = tan(value);
                break;
            case Activation::SIGMOID:
                node->value = 1.0 / (1.0 + exp(-value));
                break;
            case Activation::RELU:
                node->value = max(value,0);
                break;
            case Activation::SWISH:
                node->value = value * (1.0 / (1.0 + exp(-value)));
                break;
            case Activation::LINEAR:
                node->value = value;
                break;
            }
        }
        #endif
        #ifdef MATRIX_OPTIMIZE
        void PerceptronUpdateCalculation(
            lantern::utility::Vector<af::array>& parameters,
            lantern::utility::Vector<lantern::perceptron::Activation>& operators,
            lantern::utility::Vector<af::array>& outputs
        ){
            af::array inputs = parameters[0];
            af::array param;

            // push the input for calculate in backpropagation
            // because the input's need to calculate gradient
            outputs.push_back(inputs);
            af::array result;

            for(uint32_t i = 1; i < parameters.size(); ++i){
                
                param = parameters[i].T();
                result = af::matmul(param(af::span,af::seq(0,param.dims(1) - 2)),inputs);
                result += param.col(param.dims(1) - 1);
                result.eval();
                
                switch (operators[i - 1])
                {
                case Activation::SIGMOID:
                    result = 1 / ( 1 + af::exp(-result));
                    result.eval();
                    break;
                case Activation::RELU:
                    
                    break;
                case Activation::SWISH:
                    result = result * (1 / ( 1 + af::exp(-result)));
                    result.eval();
                    break;
                case Activation::LINEAR:
                    // do nothing
                    break;
                }
                inputs = result;
                outputs.push_back(result);
            }
            
        }
        #endif

        void FeedForward(
            #ifdef OPTIMIZE_VERSION
            lantern::utility::Vector<Perceptron *> &fix_position_node
            #endif
            #ifdef MATRIX_OPTIMIZE
            lantern::utility::Vector<af::array> &parameters
            ,lantern::utility::Vector<lantern::perceptron::Activation> &operators
            ,lantern::utility::Vector<af::array> &outputs
            #endif
        )
        {
            #ifdef OPTIMIZE_VERSION
            Perceptron *current_node = nullptr;
            for (int32_t i = fix_position_node.size() - 1; i >= 0;)
            {
                current_node = fix_position_node[i];
                PerceptronUpdateCalculation(current_node);
                --i;
            }
            #endif

            #ifdef MATRIX_OPTIMIZE
            outputs.clear();
            PerceptronUpdateCalculation(
                parameters,
                operators,
                outputs
            );
            #endif
        }

        #ifdef MATRIX_OPTIMIZE
        template <typename Optimizer>
        #endif
        void FeedForward(
            lantern::perceptron::Layer& model_layer
            #ifdef MATRIX_OPTIMIZE
            ,lantern::utility::Vector<af::array> &parameters
            ,lantern::utility::Vector<af::array> &gradient_based_parameters
            ,lantern::utility::Vector<lantern::perceptron::Activation> &operators
            ,lantern::utility::Vector<af::array> &outputs
            ,Optimizer& opt
            #endif
        )
        {

            lantern::utility::Vector<Perceptron*> fix_position_node = model_layer.GetNode();
            lantern::perceptron::Perceptron *current_node = nullptr;
            
            #ifdef MATRIX_OPTIMIZE
            uint32_t layer = 0;
            af::array temp_weight, temp_weight_gradient_based_input, vec_vel, stack_prev_grad;

            lantern::utility::Vector<double> inputs;
            lantern::utility::Vector<lantern::utility::Vector<double>> gradients,vector_velocity, stack_previous_gradient;
            #endif

            for (int32_t i = fix_position_node.size() - 1; i >= 0;)
            {
                current_node = fix_position_node[i];
                if (!current_node->IsGradientInit())
                {
                    /**
                     * create gradient tensor for current node which will feed
                     * and make sure to add one more slot for bias and check if
                     * current node has op == Activation::NOTHING which mean it was an input
                     * and no need to have bias
                     * after that set current_node->SetGradientInit(true);
                     * 
                     */
                    
                    #ifdef OPTIMIZE_VERSION
                    if(!current_node->IsGradientInit()){
                        current_node->gradient = std::move(lantern::utility::GenerateRandomNormalDVector<double>(max(current_node->total_gradient_size + (current_node->op == Activation::NOTHING? 0 : (current_node->total_gradient_size == 0? 2 : 1)), 1), 0.0f, 1.0f));
                        current_node->gradient_based_input = std::move(lantern::utility::Vector<double>(max(current_node->total_gradient_size + (current_node->op == Activation::NOTHING? 0 : (current_node->total_gradient_size == 0? 2 : 1)), 1), 1.0f));
                        
                        if(current_node->op != Activation::NOTHING){
                            current_node->gradient[current_node->total_gradient_size] = 0.0f;
                            current_node->gradient_based_input[current_node->total_gradient_size] = 0.0f;
                        }

                        current_node->SetGradientInit(true);
                    }
                    #endif

                    #ifdef MATRIX_OPTIMIZE
                    if(!current_node->IsGradientInit()){
                        current_node->gradient = std::move(lantern::utility::GenerateRandomNormalDVector<double>(current_node->parents.size() + 1, 0.0f, 1.0f));
                        current_node->gradient_based_input = std::move(lantern::utility::Vector<double>(current_node->parents.size(), 1.0f));
                        
                        current_node->vector_velocity = std::move(lantern::utility::Vector<double>(current_node->parents.size() + 1, 0.0f));
                        current_node->stack_prev_gradient = std::move(lantern::utility::Vector<double>(current_node->parents.size() + 1, 0.0f));
                        
                        current_node->SetGradientInit(true);
                        current_node->SetVectorVelocityInit(true);
                        current_node->SetPrevParamsInit(true);
                    }

                    if(current_node->op != Activation::NOTHING){
                        current_node->gradient[current_node->parents.size()] = 0.0f;
                    }
                    #endif

                    
                    #ifdef MATRIX_OPTIMIZE
                    
                    /**
                     * check if the current layer was input layer
                     */
                    if(current_node->layer == 0)
                    {
                        
                        /**
                         * push all pointer from current node which is 
                         * an input node, and set the matrix_gradient_base_input to empty
                         */
                        inputs.push_back(*current_node->value);
                        if(temp_weight.isempty()){
                            temp_weight_gradient_based_input = af::array();
                            vec_vel = af::array();
                            stack_prev_grad = af::array();
                        }

                    }else{
                        
                        if(temp_weight.isempty()){
                            /**
                             * get inputs data from utility vector
                             * then create vector input from it
                             */
                            temp_weight = af::array(inputs.size(),1,inputs.getData());
                        }

                        gradients.push_back(current_node->gradient);
                        gradients.push_back(current_node->gradient_based_input);
                        vector_velocity.push_back(current_node->vector_velocity);
                        stack_previous_gradient.push_back(current_node->stack_prev_gradient);

                        if(layer != current_node->layer){
                            layer = current_node->layer;
                            
                            operators.push_back(current_node->op);
                            parameters.push_back(temp_weight);
                            opt.vector_velocity.push_back(vec_vel);
                            opt.stack_previous_gradient.push_back(stack_prev_grad);

                            gradient_based_parameters.push_back(temp_weight_gradient_based_input);
                            
                            /**
                             * reset temp weight and temp_weight_gradient_based_input
                             */
                            temp_weight = af::array();
                            temp_weight_gradient_based_input = af::array();
                            vec_vel = af::array();
                            stack_prev_grad = af::array();
                            
                            /**
                             * set temp_weight with current gradient of node
                             * and + 1 the gradient size which the size based on parents.size()
                             * to add all weights
                             */
                            temp_weight = af::array(current_node->parents.size() + 1, 1,gradients[gradients.size()-2].getData());
                            temp_weight_gradient_based_input = af::array(current_node->parents.size(), 1,gradients[gradients.size()-1].getData());
                            vec_vel = af::array(current_node->parents.size() + 1,1, vector_velocity[vector_velocity.size()-1].getData());
                            stack_prev_grad = af::array(current_node->parents.size() + 1,1, stack_previous_gradient[stack_previous_gradient.size()-1].getData());

                        }
                        else
                        {
                            /**
                             * join current weights and bias to prev weights and bias
                             */
                            temp_weight = af::join(
                                1, 
                                temp_weight,
                                af::array(current_node->parents.size() + 1, 1,gradients[gradients.size()-2].getData())
                            );
                            temp_weight_gradient_based_input = af::join(
                                1, 
                                temp_weight_gradient_based_input,
                                af::array(current_node->parents.size(), 1,gradients[gradients.size()-1].getData())
                            );
                            vec_vel = af::join(
                                1, 
                                vec_vel,
                                af::array(current_node->parents.size() + 1, 1, vector_velocity[vector_velocity.size()-1].getData())
                            );
                            stack_prev_grad = af::join(
                                1, 
                                stack_prev_grad,
                                af::array(current_node->parents.size() + 1, 1, stack_previous_gradient[stack_previous_gradient.size()-1].getData())
                            );
                        } 
                    }
                    
                    #endif
                }

                #ifdef OPTIMIZE_VERSION
                PerceptronUpdateCalculation(
                    current_node
                );
                #endif

                --i;
            }

            
            #ifdef OPTIMIZE_VERSION
            fix_position_node[0]->gradient[0] = 1.0f;
            fix_position_node[0]->gradient[1] = 0.0f;
            #endif
            
            #ifdef MATRIX_OPTIMIZE
            parameters.push_back(temp_weight);
            gradient_based_parameters.push_back(temp_weight_gradient_based_input);
            opt.vector_velocity.push_back(vec_vel);
            opt.stack_previous_gradient.push_back(stack_prev_grad);

            PerceptronUpdateCalculation(
                parameters,
                operators,
                outputs
            );
            #endif

            /**
             * ! THIS ONLY FOR DEBUGGING
             */
            // std::cout << "All parameters : \n";
            // for(auto& g : gradients){
            //     std::cout << g << "\n";
            // }
            // std::cout << std::string(50,'=') << "\n";
            // std::cout << "input and weights : \n";
            // uint32_t j = 0;
            // for(auto& p : parameters){
            //     if(j != 0){
            //         af_print(p.T());
            //     }else{
            //         af_print(p);
            //     }
            //     ++j;
            // }
            // std::cout << std::string(50,'=') << "\n";
            // std::cout << "gradient based input : \n";
            // for(auto& gbp : gradient_based_parameters){
            //     std::cout << gbp << "\n";
            // }
            // std::cout << std::string(50,'=') << "\n";

        };


    }

}