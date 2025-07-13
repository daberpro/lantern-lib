#pragma once
#include "../pch.h"

namespace lantern {

    namespace experimental {

        namespace cnn {

            namespace feedforward {

                template <typename Optimizer = lantern::optimizer::AdaptiveMomentEstimation>
                void Initialize(
                    lantern::cnn::layer::Layer& _layer,
                    lantern::utility::Vector<af::array>& _weights,
                    lantern::utility::Vector<af::array>& _bias,
                    lantern::utility::Vector<af::array>& _prev_gradient,
                    lantern::utility::Vector<af::array>& _outputs,
                    Optimizer& _optimizer,
                    const lantern::utility::Vector<uint32_t>& _input_size
                ){
    
                    auto stack_previous_gradient_ = _optimizer.GetStackPrevGrad();
                    auto vector_velocity_ = _optimizer.GetVectorVelocity();
                    auto all_node_type_ = _layer.GetAllNodeTypeOfLayer();
                    auto all_convolve_layer_info_ = _layer.GetAllConvolveLayerInfo();
                    auto all_pooling_layer_info_ = _layer.GetAllPoolingLayerInfo();
                    auto all_layer_size_ = _layer.GetAllLayerSizes();

                    // clear all parameters and value
                    _weights.clear();
                    _outputs.clear();
                    _bias.clear();
                    _prev_gradient.clear();
                    stack_previous_gradient_.clear();
                    vector_velocity_.clear();

                    // the first outputs is input image
                    _outputs.push_back(
                        af::constant(
                            0.0f,
                            _input_size[0],
                            _input_size[1],
                            _input_size[2],
                            f64
                        )
                    );

                    uint32_t convolve_index_ = 0, pooling_index_ = 0, width_ = 0, height_ = 0, depth_ = 0, out_width_ = 0, out_height_ = 0;
                    uint32_t rest_width = 0 ,rest_height = 0 ,padding_width = 0 ,padding_height = 0;

                    for(uint32_t i = 0; i < all_layer_size_->size(); i++){

                        // get convolve info and pooling info from layer
                        lantern::cnn::layer::ConvolveLayerInfo& convolve_info_ = (*all_convolve_layer_info_)[convolve_index_];
                        lantern::cnn::layer::PoolingLayerInfo& pooling_info_ = (*all_pooling_layer_info_)[pooling_index_];

                        width_ = _outputs.back().dims(0);
                        height_ = _outputs.back().dims(1);
                        depth_ = _outputs.back().dims(2);

                        if(width_ == 0 || height_ == 0 || depth_ == 0){
                            throw std::runtime_error(std::format("Width, Height, or Depth of prev output cannot be zero at Layer [{}]\n", _outputs.size()));
                        }

                        out_width_ = (width_ - convolve_info_.kernel_size + 2 * convolve_info_.padding_size) / convolve_info_.stride_size + 1;
                        out_height_ = (height_ - convolve_info_.kernel_size + 2 * convolve_info_.padding_size) / convolve_info_.stride_size + 1;

                        // check the type of node first
                        switch((*all_node_type_)[i]){
                            case lantern::cnn::node::NodeType::CONVOLVE:

                                _weights.push_back(
                                    af::randn(
                                        convolve_info_.kernel_size, // kernel width
                                        convolve_info_.kernel_size, // kernel height
                                        convolve_info_.kernel_depth, // kernel depth
                                        (*all_layer_size_)[i] // total kernel
                                    )
                                );

                                vector_velocity_.push_back(
                                    af::constant(
                                        0.0f,
                                        convolve_info_.kernel_size, // kernel width
                                        convolve_info_.kernel_size, // kernel height
                                        convolve_info_.kernel_depth, // kernel depth
                                        (*all_layer_size_)[i], // total kernel
                                        f64
                                    )
                                );

                                stack_previous_gradient_.push_back(
                                    af::constant(
                                        0.0f,
                                        convolve_info_.kernel_size, // kernel width
                                        convolve_info_.kernel_size, // kernel height
                                        convolve_info_.kernel_depth, // kernel depth
                                        (*all_layer_size_)[i], // total kernel
                                        f64
                                    )
                                );

                                _prev_gradient.push_back(
                                    af::constant(
                                        0.0f,
                                        convolve_info_.kernel_size, // kernel width
                                        convolve_info_.kernel_size, // kernel height
                                        convolve_info_.kernel_depth, // kernel depth
                                        (*all_layer_size_)[i], // total kernel
                                        f64
                                    )
                                );

                                _outputs.push_back(
                                    af::constant(
                                        0.0f,
                                        out_width_,
                                        out_height_,
                                        (*all_layer_size_)[i],
                                        f64
                                    )
                                );

                                _bias.push_back(
                                    af::constant(
                                        0.0f,
                                        out_width_,
                                        out_height_,
                                        (*all_layer_size_)[i],
                                        f64
                                    )
                                );

                                convolve_index_++;

                            break;
                            case lantern::cnn::node::NodeType::MAX_POOL:
                                
                                rest_width = width_ % pooling_info_.size_w; // Get valid width of pooling 
                                rest_height = height_ % pooling_info_.size_h; // Get valid height of 

                                padding_width = (rest_width == 0)? 0 : (pooling_info_.size_w - rest_width); // Get the required size to create valid pooling from rest width
                                padding_height = (rest_height == 0) ? 0 : (pooling_info_.size_h - rest_height);  // Get the required size to create valid pooling from rest height
                                
                                // Output for Pooling
                                _outputs.push_back(
                                    af::constant(
                                        0.0f,
                                        ((width_ + padding_width) / pooling_info_.size_w) / pooling_info_.stride_w,
                                        ((height_ + padding_height) / pooling_info_.size_h) / pooling_info_.stride_h,
                                        depth_,
                                        f64
                                    )
                                );

                                pooling_index_++;

                            break;
                            case lantern::cnn::node::NodeType::RELU :
                                
                                _outputs.push_back(
                                    af::constant(
                                        0.0f,
                                        width_,
                                        height_,
                                        depth_,
                                        f64
                                    )
                                );

                            break;
                        }

                    }

    
                }

            }

        }

    }

}