/**
 * This file contain a perceptron calculation for feed forward
 * to use this you must defin a macro with MATRIX_OPTIMIZE or OPTIMIZE_VERSION
 * 
 */
#pragma once
#include "../pch.h"
#include "Perceptron.h"
// already define in Perceptron.h
// #include "../Headers/Vector.h"

namespace lantern
{

    namespace perceptron
    {

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
            af::array result, exp_of_result;

            for(uint32_t i = 1; i < parameters.size(); ++i){
                
                param = parameters[i].T();
                
                /**
                 * if the current activation function is not softmax
                 * then just feed the node with input without using
                 * any weights and bias
                 * 
                 */
                if(operators[i-1] != Activation::SOFTMAX){
                    result = af::matmul(param(af::span,af::seq(0,param.dims(1) - 2)),inputs);
                    result += param.col(param.dims(1) - 1);
                    result.eval();
                }else{
                    result = inputs;
                }
                
                switch (operators[i - 1])
                {
                case Activation::SIGMOID:
                    result = 1 / ( 1 + af::exp(-result));
                    result.eval();
                    break;
                case Activation::RELU:
                    result(af::where(result < 0)) = 0;
                    break;
                case Activation::SWISH:
                    result = result * (1 / ( 1 + af::exp(-result)));
                    result.eval();
                    break;
                case Activation::SOFTMAX:
                    exp_of_result = af::exp(result - af::max(result));
                    result = exp_of_result / af::sum(exp_of_result);
                    result.eval();
                    break;
                case Activation::LINEAR:
                    // do nothing
                    // because the activation function is f(x) = x
                    break;
                }
                inputs = result;
                outputs.push_back(result);
            }
            
        }
        
        /**
         * @brief Feed forward after init
         * 
         */
        void FeedForward(
            lantern::utility::Vector<af::array> &parameters, // a container which containt weights and bias
            lantern::utility::Vector<lantern::perceptron::Activation> &operators, // a container which contain activation between layers
            lantern::utility::Vector<af::array> &outputs // a container which contain outputs of previous feed forward
        )
        {
            /**
             * update calculation of each layer
             * for matrix_optimize
             * 
             * previous outputs need to be clean because
             * the update calculation will add new outputs vector
             */
            outputs.clear();
            PerceptronUpdateCalculation(
                parameters,
                operators,
                outputs
            );
        }

        template <typename Optimizer>
        /**
         * @brief Feed forward init
         * 
         */
        void FeedForward(
            lantern::perceptron::Layer& model_layer, // get layer of model
            lantern::utility::Vector<af::array> &parameters, // a container to save parameters from model such as weight and bias
            lantern::utility::Vector<af::array> &gradient_based_parameters, // a container to save gradient for computing 
            lantern::utility::Vector<lantern::perceptron::Activation> &operators, // a container to save activation function between layer
            lantern::utility::Vector<af::array> &outputs, // a container to save outpus from feed forward
            Optimizer& opt, // optimizer 
            lantern::utility::Vector<af::array> &batch_gradient // a container to save current gradient to mini batch
        )
        {

            /**
             * get fix position node from layer
             * it's will work for all type of optimization behavior
             */
            lantern::utility::Vector<Perceptron*> fix_position_node = model_layer.GetNode();
            lantern::perceptron::Perceptron *current_node = nullptr;
            
            uint32_t layer = 0;
            af::array temp_weight, temp_weight_gradient_based_input, vec_vel, stack_prev_grad;

            lantern::utility::Vector<double> inputs;
            lantern::utility::Vector<lantern::utility::Vector<double>> gradients,vector_velocity, stack_previous_gradient;
            
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
                    
                    if(!current_node->IsGradientInit()){
                        /**
                         * initalize weights using Xavier Glorot method
                         * which use to scale of acivations function and gradient
                         * with formula W_ij ~ N(0,sqrt(2/(num_in,num_out)))
                         */
                        current_node->gradient = std::move(
                            lantern::utility::GenerateRandomNormalDVector<double>(
                                current_node->parents.size() + 1, 
                                0.0f,
                                sqrt((double)2.0 /(model_layer.GetTotalNodeOnLayer(current_node->layer == 0? 0: current_node->layer - 1) + current_node->parents.size()))
                            )
                        );
                        current_node->gradient_based_input = std::move(lantern::utility::Vector<double>(current_node->parents.size(), 1.0f));
                        current_node->SetGradientInit(true);
                    }

                    if(current_node->op != Activation::NOTHING){
                        current_node->gradient[current_node->parents.size()] = 0.0f;
                    }
                    
                    
                    
                    /**
                     * check if the current layer was input layer
                     */
                    if(current_node->layer == 0)
                    {
                        
                        /**
                         * push all pointer from current node which is 
                         * an input node, and set the matrix_gradient_base_input to empty
                         */
                        inputs.push_back(1.0f);
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
                        vector_velocity.push_back(lantern::utility::Vector<double>(current_node->parents.size() + 1, 0.0f));
                        stack_previous_gradient.push_back(lantern::utility::Vector<double>(current_node->parents.size() + 1, 0.0f));

                        if(layer != current_node->layer){
                            layer = current_node->layer;
                            
                            operators.push_back(current_node->op);
                            parameters.push_back(temp_weight);
                            opt.vector_velocity.push_back(vec_vel);
                            opt.stack_previous_gradient.push_back(stack_prev_grad);
                            
                            gradient_based_parameters.push_back(temp_weight_gradient_based_input);
                            
                            // this only for init mini batch
                            batch_gradient.push_back(af::constant(0.0f,temp_weight.dims(0),temp_weight.dims(1)));

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
                    
                }

                --i;
            }
            
            parameters.push_back(temp_weight);
            gradient_based_parameters.push_back(temp_weight_gradient_based_input);
            opt.vector_velocity.push_back(vec_vel);
            opt.stack_previous_gradient.push_back(stack_prev_grad);
            
            // this only for init mini batch
            batch_gradient.push_back(af::constant(0.0f,temp_weight.dims(0),temp_weight.dims(1)));           

            PerceptronUpdateCalculation(
                parameters,
                operators,
                outputs
            );
            
        };


    }

}