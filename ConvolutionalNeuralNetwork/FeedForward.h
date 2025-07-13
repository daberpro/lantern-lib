#pragma once
#include "../pch.h"
#include "Node.h"
#include "Layer.h"
#include "../Headers/Vector.h"
#include "../Headers/Function.h"
#include "../FeedForwardNetwork/Optimizer/Optimizer.h"

namespace lantern {
    namespace cnn {
        namespace feedforward {

            void FeedForward(
                lantern::cnn::layer::Layer& _layer,
                lantern::utility::Vector<af::array>& _weights,
                lantern::utility::Vector<af::array>& _bias,
                lantern::utility::Vector<af::array>& _outputs
            ){

                auto all_node_type = _layer.GetAllNodeTypeOfLayer();
                auto all_convolve_layer_info = _layer.GetAllConvolveLayerInfo();
                auto all_pooling_layer_info = _layer.GetAllPoolingLayerInfo();
                auto all_layer_size = _layer.GetAllLayerSizes();

                uint32_t convolve_index_ = 0, pooling_index_ = 0, params_index_ = 0;
                af::array outputs_, derivative_;


                /*
                for (auto w : _weights) {
                    std::print("Weights: \n {}", w);
                }*/

                for(uint32_t i = 0; i < all_layer_size->size(); i++){
                    
                    lantern::cnn::layer::ConvolveLayerInfo& convolve_info = (*all_convolve_layer_info)[convolve_index_];
                    lantern::cnn::layer::PoolingLayerInfo& max_pool_info = (*all_pooling_layer_info)[pooling_index_];
                    
                    switch((*all_node_type)[i]){
                        case lantern::cnn::node::NodeType::CONVOLVE:

                            for (uint32_t j = 0; j < (*all_layer_size)[i]; j++) {
                                if (outputs_.isempty()) {
                                    outputs_ = af::convolve2NN(
                                        _outputs[i],
                                        _weights[params_index_],
                                        af::dim4(convolve_info.stride_size, convolve_info.stride_size),
                                        af::dim4(convolve_info.padding_size, convolve_info.padding_size),
                                        af::dim4(1, 1)
                                    ) + _bias[params_index_];
                                }
                                else {
                                    outputs_ = af::join(
                                        2,
                                        outputs_,
                                        af::convolve2NN(
                                            _outputs[i],
                                            _weights[params_index_],
                                            af::dim4(convolve_info.stride_size, convolve_info.stride_size),
                                            af::dim4(convolve_info.padding_size, convolve_info.padding_size),
                                            af::dim4(1, 1)
                                        ) + _bias[params_index_]
                                    );
                                }
                                params_index_++;
                            }

                            
                        break;
                        case lantern::cnn::node::NodeType::RELU:

                            outputs_ = lantern::activation::ReLU(_outputs[i]);
                            
                        break;
                        case lantern::cnn::node::NodeType::FLATTEN:

                            outputs_ = _outputs[i];
                            outputs_ = af::moddims(outputs_,outputs_.dims(0) * outputs_.dims(1) * outputs_.dims(2),1,1);

                        break;
                        case lantern::cnn::node::NodeType::MAX_POOL:

                            
                            // Get pooling width and height
                            int32_t width = _outputs[i].dims(0);
                            int32_t height = _outputs[i].dims(1);
                            int32_t depth = _outputs[i].dims(2);
                            int32_t rest_width = width % max_pool_info.size_w; // Get valid width of pooling 
                            int32_t rest_height = height % max_pool_info.size_h; // Get valid height of 

                            int32_t padding_width = (rest_width == 0) ? 0 : (max_pool_info.size_w - rest_width); // Get the required size to create valid pooling from rest width
                            int32_t padding_height = (rest_height == 0) ? 0 : (max_pool_info.size_h - rest_height);  // Get the required size to create valid pooling from rest height

                            
                            outputs_ = af::pad(
                                _outputs[i],
                                af::dim4(0, 0, 0, 0),
                                af::dim4(padding_width, padding_height, 0, 0),
                                AF_PAD_ZERO
                            );

                            outputs_ = lantern::pooling::MaxPool(
                                outputs_,
                                max_pool_info.size_h,
                                max_pool_info.size_w
                            );

                            pooling_index_++;
                            
                        break;
                    }

                    _outputs[i + 1] = outputs_;

                    outputs_ = af::array();
                }

    
            }

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

                auto stack_previous_gradient = _optimizer.GetStackPrevGrad();
                auto vector_velocity = _optimizer.GetVectorVelocity();
                auto all_node_type = _layer.GetAllNodeTypeOfLayer();
                auto all_convolve_layer_info = _layer.GetAllConvolveLayerInfo();
                auto all_pooling_layer_info = _layer.GetAllPoolingLayerInfo();
                auto all_layer_size = _layer.GetAllLayerSizes();

                _weights.clear();
                _outputs.clear();
                _bias.clear();
                _prev_gradient.clear();
                stack_previous_gradient.clear();
                vector_velocity.clear();

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

                uint32_t kernel_index_ = 0;

                // Get pooling width and height
                uint32_t width;
                uint32_t height;
                uint32_t depth;

                
                for(uint32_t i = 0; i < (*all_layer_size).size() - 1; i++){

                    const auto& [kernel_size, padding_size, stride_size, kernel_depth] = (*all_convolve_layer_info)[kernel_index_];
                    const auto& [pool_w, pool_h, _stride_w, _stride_h] = (*all_pooling_layer_info)[kernel_index_];

                    if(_outputs.size() > 0){
                        width = _outputs.back().dims(0);
                        height = _outputs.back().dims(1);
                        depth = _outputs.back().dims(2);
                    }else{
                        width = _input_size[0];
                        height = _input_size[1];
                        depth = _input_size[2];
                    }

                    if(width == 0 || height == 0 || depth == 0){
                        throw std::runtime_error(std::format("Width, Height, or Depth of prev output cannot be zero at Layer [{}]\n", _outputs.size()));
                    }

                    switch((*all_node_type)[i]){
                        case lantern::cnn::node::NodeType::CONVOLVE:

                            // check if the kernel size is bigger than input
                            if ((height < kernel_size || width < kernel_size) && _outputs.size() > 0) {
                                throw std::runtime_error(std::format("Kernel size are bigger than signal size at Layer [{}]\n", _outputs.size()));
                            }
                            // check if the kernel depth are not same from input
                            if (depth != kernel_depth) {
                                throw std::runtime_error(std::format("Kernel depth and signal depth must be same at Layer [{}]\n", _outputs.size()));
                            }


                            for(uint32_t j = 0; j < (*all_layer_size)[i]; j++){
                                _weights.push_back(
                                    af::randn(
                                        kernel_size, // kernel width
                                        kernel_size, // kernel height
                                        kernel_depth, // total kernel
                                        f64
                                    )
                                );
                                stack_previous_gradient.push_back(
                                    af::constant(
                                        0.0f,
                                        kernel_size, // kernel width
                                        kernel_size, // kernel height
                                        kernel_depth, // total kernel
                                        f64
                                    )
                                );
                                _prev_gradient.push_back(
                                    af::constant(
                                        0.0f,
                                        kernel_size, // kernel width
                                        kernel_size, // kernel height
                                        kernel_depth, // total kernel
                                        f64
                                    )
                                );
                                vector_velocity.push_back(
                                    af::constant(
                                        0.0f,
                                        kernel_size, // kernel width
                                        kernel_size, // kernel height
                                        kernel_depth, // total kernel
                                        f64
                                    )
                                );

                                /**
                                 * The bias width and height are same with the outputs of convolution
                                 * W_out = (W_in - F + 2P)/S + 1
                                 * H_out = (H_in - F + 2P)/S + 1
                                 */
                                _bias.push_back(
                                    af::constant(
                                        0.0f,
                                        (width - kernel_size + 2 * padding_size) / stride_size + 1,
                                        (height - kernel_size + 2 * padding_size) / stride_size + 1,
                                        f64
                                    )
                                );
                                

                                /**
                                 * the input size and output size
                                 * was depend on kernel volume
                                 */
                                lantern::init::XavierNormInit(
                                    kernel_size * kernel_size * kernel_depth,
                                    (*all_convolve_layer_info)[kernel_index_ + 1].kernel_size * 
                                    (*all_convolve_layer_info)[kernel_index_ + 1].kernel_size * 
                                    (*all_convolve_layer_info)[kernel_index_ + 1].kernel_depth,
                                    _weights.back()
                                );
                                
                            }

                            // Output for Convolution
                            _outputs.push_back(
                                af::constant(
                                    0.0f,
                                    (width - kernel_size + 2 * padding_size) / stride_size + 1,
                                    (height - kernel_size + 2 * padding_size) / stride_size + 1,
                                    (*all_layer_size)[i],
                                    f64
                                )
                            );

                            kernel_index_++;
                        break;
                        case lantern::cnn::node::NodeType::RELU:
                            
                            _outputs.push_back(
                                af::constant(
                                    0.0f,
                                    width,
                                    height,
                                    depth,
                                    f64
                                )
                            );
                            
                        break;
                        case lantern::cnn::node::NodeType::FLATTEN:
                            
                            _outputs.push_back(
                                af::constant(
                                    0.0f,
                                    width * height * depth,
                                    1,
                                    1,
                                    f64
                                )
                            );

                        break;
                        /**
                         * WARNING! never put another case below this case
                         * it will cause compilation error because the declaration inside this case
                         */
                        case lantern::cnn::node::NodeType::MAX_POOL:

                            int32_t rest_width = width % pool_w; // Get valid width of pooling 
                            int32_t rest_height = height % pool_h; // Get valid height of 

                            int32_t padding_width = (rest_width == 0)? 0 : (pool_w - rest_width); // Get the required size to create valid pooling from rest width
                            int32_t padding_height = (rest_height == 0) ? 0 : (pool_h - rest_height);  // Get the required size to create valid pooling from rest height
                            
                            // Output for Pooling
                            _outputs.push_back(
                                af::constant(
                                    0.0f,
                                    (width + padding_width) / pool_w,
                                    (height + padding_height) / pool_h,
                                    depth,
                                    f64
                                )
                            );
                        break;
                    }

                   
                }

                const auto& [kernel_size, padding_size, stride_size, kernel_depth] = all_convolve_layer_info->back();
                const auto& [pool_w, pool_h, _stride_w, _stride_h] = all_pooling_layer_info->back();

                if (_bias.size() > 0) {
                    width = _bias.back().dims(0);
                    height = _bias.back().dims(1);
                    depth = _bias.back().dims(2);
                }
                else {
                    width = _input_size[0];
                    height = _input_size[1];
                    depth = _input_size[2];
                }

                if(width == 0 || height == 0 || depth == 0){
                    throw std::runtime_error(std::format("Width, Height, or Depth of prev output cannot be zero at Layer [{}]\n", _outputs.size()));
                }

                /*
                * Kernel Width and Height are same 
                * because lantern will assume the kernel width height always same and the number of them is odd
                */
                switch (all_node_type->back()) {
                    case lantern::cnn::node::NodeType::CONVOLVE:

                        // check if the kernel size is bigger than input
                        if ((height < kernel_size || width < kernel_size) && _outputs.size() > 0) {
                            throw std::runtime_error(std::format("Kernel size are bigger than signal size at Layer [{}]\n",_outputs.size()));
                        }

                        // check if the kernel depth are not same from input
                        if (depth != all_convolve_layer_info->back().kernel_depth) {
                            throw std::runtime_error(std::format("Kernel depth and signal depth must be same at Layer [{}]\n", _outputs.size()));
                        }

                        for (uint32_t j = 0; j < all_layer_size->back(); j++) {
                            _weights.push_back(
                                af::randn(
                                    kernel_size, // kernel width
                                    kernel_size, // kernel height
                                    kernel_depth, // total kernel
                                    f64
                                )
                            );
                            stack_previous_gradient.push_back(
                                af::constant(
                                    0.0f,
                                    kernel_size, // kernel width
                                    kernel_size, // kernel height
                                    kernel_depth, // total kernel
                                    f64
                                )
                            );
                            _prev_gradient.push_back(
                                af::constant(
                                    0.0f,
                                    kernel_size, // kernel width
                                    kernel_size, // kernel height
                                    kernel_depth, // total kernel
                                    f64
                                )
                            );
                            vector_velocity.push_back(
                                af::constant(
                                    0.0f,
                                    kernel_size, // kernel width
                                    kernel_size, // kernel height
                                    kernel_depth, // total kernel
                                    f64
                                )
                            );

                            /**
                             * The bias width and height are same with the outputs of convolution
                             * W_out = (W_in - F + 2P)/S + 1
                             * H_out = (H_in - F + 2P)/S + 1
                             */
                            _bias.push_back(
                                af::constant(
                                    0.0f,
                                    (width - kernel_size + 2 * padding_size) / stride_size + 1,
                                    (height - kernel_size + 2 * padding_size) / stride_size + 1,
                                    f64
                                )
                            );


                            /**
                             * the input size and output size
                             * was depend on kernel volume
                             */
                            lantern::init::XavierNormInit(
                                kernel_size * kernel_size * kernel_depth,
                                (*all_convolve_layer_info)[kernel_index_ + 1].kernel_size *
                                (*all_convolve_layer_info)[kernel_index_ + 1].kernel_size *
                                (*all_convolve_layer_info)[kernel_index_ + 1].kernel_depth,
                                _weights.back()
                            );

                        }

                        // Output for Convolution
                        _outputs.push_back(
                            af::constant(
                                0.0f,
                                (width - kernel_size + 2 * padding_size) / stride_size + 1,
                                (height - kernel_size + 2 * padding_size) / stride_size + 1,
                                all_layer_size->back(),
                                f64
                            )
                        );

                        kernel_index_++;
                    break;
                    case lantern::cnn::node::NodeType::RELU:
                        _outputs.push_back(
                            af::constant(
                                0.0f,
                                width,
                                height,
                                depth,
                                f64
                            )
                        );
                    break;
                    case lantern::cnn::node::NodeType::FLATTEN:
                        _outputs.push_back(
                            af::constant(
                                0.0f,
                                width * height * depth,
                                1,
                                1,
                                f64
                            )
                        );
                    break;
                    /**
                     * WARNING! never put another case below this case
                     * it will cause compilation error because the declaration inside this case
                     */
                    case lantern::cnn::node::NodeType::MAX_POOL:

                        int32_t rest_width = width % pool_w; // Get valid width of pooling 
                        int32_t rest_height = height % pool_h; // Get valid height of 

                        int32_t padding_width = (rest_width == 0) ? 0 : (pool_w - rest_width); // Get the required size to create valid pooling from rest width
                        int32_t padding_height = (rest_height == 0) ? 0 : (pool_h - rest_height);  // Get the required size to create valid pooling from rest height

                        // Output for Pooling
                        _outputs.push_back(
                            af::constant(
                                0.0f,
                                (width + padding_width) / pool_w,
                                (height + padding_height) / pool_h,
                                depth,
                                f64
                            )
                        );
                    break;
                }
                
    
        
            }
        }

    }
}