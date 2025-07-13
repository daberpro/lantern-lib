#pragma once
#include "../pch.h"
#include "Node.h"
#include "Layer.h"
#include "../Headers/Function.h"
#include "../Headers/Vector.h"

namespace lantern {

	namespace cnn {

		namespace backprop {

			template <typename Optimizer>
            void Backpropagate(
                lantern::cnn::layer::Layer& _layer,
                lantern::utility::Vector<af::array>& _weights,
                lantern::utility::Vector<af::array>& _bias,
                lantern::utility::Vector<af::array>& _prev_gradient,
                lantern::utility::Vector<af::array>& _outputs,
                Optimizer& _optimizer,
                const uint32_t _batch_size
            ){

				double batch_size = static_cast<double>(_batch_size);
                lantern::utility::Vector<uint32_t>* all_layer_sizes = _layer.GetAllLayerSizes();
                lantern::utility::Vector<lantern::cnn::node::NodeType>* all_layer_type = _layer.GetAllNodeTypeOfLayer();
                lantern::utility::Vector<lantern::cnn::layer::ConvolveLayerInfo>* all_convolve_layer = _layer.GetAllConvolveLayerInfo();
                lantern::utility::Vector<lantern::cnn::layer::PoolingLayerInfo>* all_pooling_layer = _layer.GetAllPoolingLayerInfo();

                af::array output_, prev_output_, gradient_, gradient_weight_, gradient_bias_, weights_, bias_;
                
                uint32_t convolve_index_ = all_convolve_layer->size() - 1;
                uint32_t pooling_index_ = all_pooling_layer->size() - 1;
                
                for(uint32_t current_layer = (*all_layer_sizes).size() - 1; current_layer > 0; current_layer--){
                    
                    lantern::cnn::layer::ConvolveLayerInfo& layer_info_ = (*all_convolve_layer)[convolve_index_];
                    convolve_index_ *= (*all_layer_sizes)[current_layer];
                    prev_output_ = _outputs[current_layer];
                    output_ = _outputs[current_layer - 1];
                    weights_ = _weights[convolve_index_];
                    bias_ = _bias[convolve_index_];
                    
                    switch((*all_layer_type)[current_layer]){
                        case lantern::cnn::node::NodeType::CONVOLVE:

                            std::println("{}", _weights);
                            std::println("Input {}\n weights {}\n Out {}", _outputs[current_layer - 2], weights_, _outputs[current_layer - 1]);
                            gradient_ = af::convolve2GradientNN(
                                af::constant(1.0f, output_.dims()),
                                _outputs[current_layer - 2],
                                weights_,
                                _outputs[current_layer - 1],
                                layer_info_.stride_size,
                                layer_info_.padding_size,
                                af::dim4(0,0,0,0),
                                AF_CONV_GRADIENT_FILTER
                            );

                            std::println("Result {}", gradient_);
                            
                            exit(0);
                            convolve_index_--;
                        break;
                        case lantern::cnn::node::NodeType::MAX_POOL:
                            
                        break;
                    }


                }
    
                // for(uint32_t current_layer = (*all_layer_sizes).size() - 1; current_layer > 0; current_layer--){
                    
                //     prev_output = _outputs[current_layer];
                //     output = _outputs[current_layer - 1];
                //     af::array& parameters = _parameters[current_layer - 1];
                //     weights = parameters.cols(0,parameters.dims(1) - 2);
                    
                //     switch((*all_layer_type)[current_layer]){
                //         case lantern::cnn::node::NodeType::LINEAR:
                //             gradient = lantern::derivative::Linear(prev_output);
                //         break;
                //     }
    
                //     gradient *= _prev_gradient[current_layer];
                //     gradient.eval();
                //     gradient_weight = af::matmul(gradient,output.T());
    
                //     gradient_weight.eval();
                //     gradient_bias = gradient;
    
                //     all_gradient = af::join(
                //         1,
                //         gradient_weight,
                //         gradient_bias
                //     );
    
                //     uint32_t opt_index = current_layer - 1;
                //     af::array delta = _optimizer.GetDelta(all_gradient,opt_index);
                //     parameters -= delta;
                //     parameters.eval();
                    
                //     _prev_gradient[current_layer - 1] = af::matmul(
                //         parameters(af::span,af::seq(parameters.dims(1) - 1)).T(),
                //         gradient
                //     );
    
                // }

			}

		}

	}

}
