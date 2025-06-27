#pragma once
#include "../pch.h"
#include "Function.h"
#include "Layering.h"
#include "Node.h"
#include "Regularization.h"

namespace lantern{

    namespace ffn {
        
        namespace backprop {
    
            /**
             * @brief Backpropagate thorught model
             * 
             * @tparam Optimizer 
             * @tparam RegularizationFunction 
             * @param _layer Layer of model
             * @param _parameters Stack of weights ans bias
             * @param _prev_gradient Stack of gradient need to compute all weights and bias
             * @param _outputs Stack of output
             * @param _optimizer Optimizer for model
             * @param _batch_size Batch size
             */
            template <typename Optimizer>
            void Backpropagate(
                lantern::ffn::layer::Layer& _layer,
                lantern::utility::Vector<af::array>& _parameters,
                lantern::utility::Vector<af::array>& _prev_gradient,
                lantern::utility::Vector<af::array>& _outputs,
                Optimizer& _optimizer,
                const uint32_t _batch_size
            ){
    
                double batch_size = static_cast<double>(_batch_size);
                lantern::utility::Vector<uint32_t>* all_layer_sizes = _layer.GetAllLayerSizes();
                lantern::utility::Vector<lantern::node::NodeType>* all_layer_type = _layer.GetAllNodeTypeOfLayer();
                af::array output, prev_output, gradient, gradient_weight, gradient_bias, all_gradient, weights;
    
                for(uint32_t current_layer = (*all_layer_sizes).size() - 1; current_layer > 0; current_layer--){
                    
                    prev_output = _outputs[current_layer];
                    output = _outputs[current_layer - 1];
                    af::array& parameters = _parameters[current_layer - 1];
                    weights = parameters.cols(0,parameters.dims(1) - 2);
                    
                    switch((*all_layer_type)[current_layer]){
                        case lantern::node::NodeType::LINEAR:
                            gradient = lantern::derivative::Linear(prev_output);
                        
                        break;
                        case lantern::node::NodeType::SIGMOID:
                            gradient = lantern::derivative::Sigmoid(prev_output);
                        
                        break;
                        case lantern::node::NodeType::RELU:
                            // gradient = lantern::derivative::ReLU(prev_output);
                        
                        break;
                        case lantern::node::NodeType::TANH:
                            // gradient = lantern::derivative::TanH(prev_output);
                        
                        break;
                        case lantern::node::NodeType::SWISH:
                            gradient = lantern::derivative::Swish(prev_output);
                        
                        break;
                    }
    
                    gradient *= _prev_gradient[current_layer];
                    gradient.eval();
                    gradient_weight = af::matmul(gradient,output.T());
    
                    gradient_weight.eval();
                    gradient_bias = gradient;
    
                    all_gradient = af::join(
                        1,
                        gradient_weight,
                        gradient_bias
                    );
    
                    uint32_t opt_index = current_layer - 1;
                    af::array delta = _optimizer.GetDelta(all_gradient,opt_index);
                    parameters -= delta;
                    parameters.eval();
                    
                    _prev_gradient[current_layer - 1] = af::matmul(
                        parameters(af::span,af::seq(parameters.dims(1) - 1)).T(),
                        gradient
                    );
    
                }
    
            }
    
    
            template <typename Optimizer>
            void Backpropagate(
                lantern::ffn::layer::Layer& _layer,
                lantern::utility::Vector<af::array>& _parameters,
                lantern::utility::Vector<af::array>& _prev_gradient,
                lantern::utility::Vector<af::array>& _outputs,
                Optimizer& _optimizer
            ){
    
                lantern::utility::Vector<uint32_t>* all_layer_sizes = _layer.GetAllLayerSizes();
                lantern::utility::Vector<lantern::node::NodeType>* all_layer_type = _layer.GetAllNodeTypeOfLayer();
                af::array output, prev_output, gradient, gradient_weight, gradient_bias, all_gradient;
    
                for(uint32_t current_layer = (*all_layer_sizes).size() - 1; current_layer > 0; current_layer--){
                    
                    prev_output = _outputs[current_layer];
                    output = _outputs[current_layer - 1];
                    af::array& parameters = _parameters[current_layer - 1];
                   
                    switch((*all_layer_type)[current_layer]){
                        case lantern::node::NodeType::LINEAR:
                            gradient = lantern::derivative::Linear(prev_output);
                        
                        break;
                        case lantern::node::NodeType::SIGMOID:
                            gradient = lantern::derivative::Sigmoid(prev_output);
                        
                        break;
                        case lantern::node::NodeType::RELU:
                            // gradient = lantern::derivative::ReLU(prev_output);
                        
                        break;
                        case lantern::node::NodeType::TANH:
                            // gradient = lantern::derivative::TanH(prev_output);
                        
                        break;
                        case lantern::node::NodeType::SWISH:
                            gradient = lantern::derivative::Swish(prev_output);
                        
                        break;
                    }
    
                    gradient *= _prev_gradient[current_layer];
                    gradient.eval();
                    gradient_weight = af::matmul(gradient,output.T());
    
                    gradient_weight.eval();
                    gradient_bias = gradient;
    
                    all_gradient = af::join(
                        1,
                        gradient_weight,
                        gradient_bias
                    );
    
                    uint32_t opt_index = current_layer - 1;
                    af::array delta = _optimizer.GetDelta(all_gradient,opt_index);
                    parameters -= delta;
                    parameters.eval();
                    
                    _prev_gradient[current_layer - 1] = af::matmul(
                        parameters(af::span,af::seq(parameters.dims(1) - 1)).T(),
                        gradient
                    );
    
                }
    
            }
    
    
        }
    }

}