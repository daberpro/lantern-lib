#pragma once
#include "../pch.h"
#include "../Headers/Function.h"
#include "../Headers/Vector.h"
#include "CNNNode.h"
#include "CNNLayer.h"
#include "CNNOptimizer/CNNOptimizer.h"

namespace lantern
{

    namespace cnn
    {

        namespace backprop
        {

            /**
             * @brief Backpropagation for CNN parameters
             * @tparam Optimizer 
             * @param _layer 
             * @param _weights 
             * @param _bias 
             * @param (*prev_gradient_) 
             * @param _outputs 
             * @param _optimizer 
             * @param _batch_size 
             * @ingroup LanternBackprop
             */
            template <typename Optimizer = lantern::cnn::optimizer::GradientDescent>
            inline void Backpropagate(
                lantern::cnn::layer::Layer& _layer,
                Optimizer &_optimizer,
                const uint32_t _batch_size
            ){

                double batch_size = static_cast<double>(_batch_size);
                auto all_layer_sizes = _layer.GetAllLayerSizes();
                auto all_layer_type = _layer.GetAllNodeTypeOfLayer();
                auto all_convolve_layer = _layer.GetAllConvolveLayerInfo();
                auto all_pooling_layer = _layer.GetAllPoolingLayerInfo();
                auto pooling_modification_input_result = _layer.GetPoolingModificationInputResult();

                auto weights_ = _layer.GetWeights();
                auto bias_ = _layer.GetBias();
                auto prev_gradient_ = _layer.GetPrevGradient();
                auto outputs_ = _layer.GetOutputs();

                auto batch_norm_params_ = _layer.GetBatchNormParams();
                auto batch_norm_derivative_params_ = _layer.GetBatchNormDerivativeParams();

                af::array output_, prev_output_, gradient_, gradient_weight_, gradient_bias_, gradient_out_;

                int convolve_index_ = all_convolve_layer->size() - 1;
                int pooling_index_ = all_pooling_layer->size() - 1;
                int batch_norm_index_ = batch_norm_params_->size() - 1;

                for (int i = all_layer_sizes->size() - 1; i > 0; i--)
                {

                    switch ((*all_layer_type)[i])
                    {
                        case lantern::cnn::node::NodeType::CONVOLVE:
                        {
                            lantern::cnn::layer::ConvolveLayerInfo &convolve_info = (*all_convolve_layer)[convolve_index_];
                            gradient_weight_ = af::convolve2GradientNN(
                                (*prev_gradient_)[i],
                                (*outputs_)[i],
                                (*weights_)[convolve_index_],
                                (*outputs_)[i + 1],
                                convolve_info.stride,
                                convolve_info.padding,
                                af::dim4(1,1,0,0),
                                AF_CONV_GRADIENT_FILTER
                            );
                            
                            gradient_ = af::convolve2GradientNN(
                                (*prev_gradient_)[i],
                                (*outputs_)[i],
                                (*weights_)[convolve_index_],
                                (*outputs_)[i + 1],
                                convolve_info.stride,
                                convolve_info.padding,
                                af::dim4(1,1,0,0),
                                AF_CONV_GRADIENT_DATA
                            );

                            gradient_weight_ = gradient_weight_.as(f64);
                            gradient_bias_ = (*prev_gradient_)[i];
                            
                            gradient_weight_ /= batch_size;
                            gradient_bias_ /= batch_size;

                            gradient_weight_.eval();
                            gradient_bias_.eval();
                            
                            gradient_bias_ = af::sum(gradient_bias_,1);
                            gradient_bias_ = af::sum(gradient_bias_,0);
                            
                            // get gradient from optimizer
                            std::pair<af::array,af::array> gradient_optimized_ = _optimizer.GetDelta(
                                gradient_weight_,
                                gradient_bias_, // because the derivative of bias just 1 we can just pass the prev gradient directly
                                convolve_index_
                            );

                            (*weights_)[convolve_index_] -=  gradient_optimized_.first; // weights is first
                            (*bias_)[convolve_index_] -= gradient_optimized_.second;

                            (*weights_)[convolve_index_].eval();
                            (*bias_)[convolve_index_].eval();

                            if (convolve_index_ - 1 >= 0) {
                                convolve_index_--;
                            }

                            break;
                        }
                        case lantern::cnn::node::NodeType::RELU:
                        {
                            gradient_ = ((*outputs_)[i] > 0).as(f64);
                            gradient_ *= (*prev_gradient_)[i];
                            gradient_.eval();
                            break;
                        }
                        case lantern::cnn::node::NodeType::SWISH:
                        {
                            gradient_ = lantern::derivative::Swish((*outputs_)[i]);
                            gradient_ *= (*prev_gradient_)[i];
                            gradient_.eval();
                            break;
                        }
                        case lantern::cnn::node::NodeType::SIGMOID:
                        {
                            gradient_ = lantern::derivative::Sigmoid((*outputs_)[i]);
                            gradient_ *= (*prev_gradient_)[i];
                            gradient_.eval();
                            break;
                        }
                        case lantern::cnn::node::NodeType::AVG_POOL:{

                            lantern::cnn::layer::PoolingLayerInfo& pool_info_ = (*all_pooling_layer)[pooling_index_];
                            af::array& modification_input_ = (*pooling_modification_input_result)[pooling_index_];
                            gradient_ = lantern::derivative::AvgPoolWithStride(
                                (*outputs_)[i],
                                modification_input_,
                                pool_info_.size_h,
                                pool_info_.size_w,
                                pool_info_.stride,
                                (*prev_gradient_)[i]
                            );
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
                                (*outputs_)[i],
                                modification_input_,
                                pool_info_.size_h,
                                pool_info_.size_w,
                                pool_info_.stride,
                                (*prev_gradient_)[i]
                            );
                            if(pooling_index_ - 1 >= 0){
                                pooling_index_--;
                            }
                            break;

                        }
                        case lantern::cnn::node::NodeType::FLATTEN:
                        {
                            gradient_ = af::constant(1.0f,(*outputs_)[i].dims(),f64);
                            gradient_ *= (*prev_gradient_)[i];
                            gradient_.eval();
                            break;
                        }
                        case lantern::cnn::node::NodeType::BATCH_NORM:
                        {
                            af::array& batch_norm_param = (*_layer.GetBatchNormParams())[batch_norm_index_];
                            af::array& batch_norm_derivative_param = (*_layer.GetBatchNormDerivativeParams())[batch_norm_index_];

                            gradient_ = (*prev_gradient_)[i];
                            batch_norm_param -= _optimizer.GetDeltaBatchNorm(batch_norm_derivative_param / batch_size, batch_norm_index_);
                            
                            if(batch_norm_index_ - 1 >= 0) {
                                batch_norm_index_--;
                            }
                            break;
                        }
                    }

                    (*prev_gradient_)[i - 1] = gradient_;

                }
            }

        }

    }

}
