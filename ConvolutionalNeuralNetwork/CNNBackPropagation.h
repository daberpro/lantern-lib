#pragma once
#include "../pch.h"
#include "../Headers/Function.h"
#include "../Headers/Vector.h"
#include "CNNNode.h"
#include "CNNLayer.h"

namespace lantern
{

    namespace cnn
    {

        namespace backprop
        {

            template <typename Optimizer>
            void Backpropagate(
                lantern::cnn::layer::Layer &_layer,
                lantern::utility::Vector<af::array> &_weights,
                lantern::utility::Vector<af::array> &_bias,
                lantern::utility::Vector<af::array> &_prev_gradient,
                lantern::utility::Vector<af::array> &_outputs,
                Optimizer &_optimizer,
                const uint32_t _batch_size)
            {

                double batch_size = static_cast<double>(_batch_size);
                auto all_layer_sizes = _layer.GetAllLayerSizes();
                auto all_layer_type = _layer.GetAllNodeTypeOfLayer();
                auto all_convolve_layer = _layer.GetAllConvolveLayerInfo();
                auto all_pooling_layer = _layer.GetAllPoolingLayerInfo();
                auto pooling_modification_input_result = _layer.GetPoolingModificationInputResult();

                af::array output_, prev_output_, gradient_, gradient_weight_, gradient_bias_, gradient_out_, weights_, bias_;

                uint32_t convolve_index_ = all_convolve_layer->size() - 1;
                uint32_t pooling_index_ = all_pooling_layer->size() - 1;
                uint32_t flatten_index = _prev_gradient.size() - 1;

                std::println("{}",(*all_layer_type));

                for (uint32_t i = all_layer_sizes->size() - 1; i > 0; i--)
                {

                    lantern::cnn::layer::ConvolveLayerInfo &convolve_info = (*all_convolve_layer)[convolve_index_];

                    switch ((*all_layer_type)[i])
                    {
                    case lantern::cnn::node::NodeType::CONVOLVE:
                    {

                        // std::println(
                        //     "kernel depth {}, kernel size {}, padding {}, stride {}",
                        //     convolve_info.kernel_depth,
                        //     convolve_info.kernel_size,
                        //     convolve_info.padding,
                        //     convolve_info.stride
                        // );

                        // std::println("{}", _outputs[i]);

                        if (convolve_index_ - 1 >= 0) {
                            convolve_index_--;
                        }

                        break;
                    }
                    case lantern::cnn::node::NodeType::RELU:
                    {
                        gradient_ = (_outputs[i] > 0).as(f64);
                        break;
                    }
                    case lantern::cnn::node::NodeType::AVG_POOL:{
                        if(pooling_index_ - 1 >= 0){
                            pooling_index_--;
                        }
                        break;
                    }
                    case lantern::cnn::node::NodeType::MAX_POOL:
                    {

                        lantern::cnn::layer::PoolingLayerInfo& pool_info_ = (*all_pooling_layer)[pooling_index_];
                        af::array& modification_input_ = (*pooling_modification_input_result)[pooling_index_];
                        gradient_ = lantern::derivative::MaxPoolWithStride(
                            _outputs[i],
                            modification_input_,
                            pool_info_.size_h,
                            pool_info_.size_w,
                            pool_info_.stride
                        );
                        if(pooling_index_ - 1 >= 0){
                            pooling_index_--;
                        }
                        break;

                    }
                    case lantern::cnn::node::NodeType::FLATTEN:
                    {
                        gradient_ = af::moddims(_prev_gradient[flatten_index],_outputs[i].dims());
                        if(flatten_index - 1 >= 0){
                            flatten_index--;
                        }
                        break;
                    }
                    }
                }
            }

        }

    }

}
