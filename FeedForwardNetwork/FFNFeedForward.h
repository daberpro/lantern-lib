#pragma once
#include "../Headers/Vector.h"
#include "../pch.h"
#include "../Headers/Function.h"
#include "../Headers/Initialize.h"
#include "FFNNode.h"
#include "FFNLayer.h"
#include "FFNOptimizer/FFNOptimizer.h"

/**
 * @defgroup LanternFeedForward FeedForward function for lantern feed
 */

namespace lantern
{

    namespace ffn
    {

        namespace feedforward
        {

            /**
             * @brief Feed Forward
             * @param _layer 
             */
            inline void FeedForward(
                lantern::ffn::layer::Layer &_layer
            )
            {

                af::array parameters_from_layer_, prev_output_, current_output_, weight_, bias_;
                lantern::utility::Vector<uint32_t> *all_layer_sizes_ = _layer.GetAllLayerSizes();
                lantern::utility::Vector<lantern::ffn::node::NodeType> *all_layer_type_ = _layer.GetAllNodeTypeOfLayer();

                auto* parameters_ = _layer.GetParameters();
                auto* outputs_ = _layer.GetOutputs();
                uint32_t all_layer_size_ = (*all_layer_sizes_).size();

                // set current layer to start at 1 because in index 0 was input
                for (uint32_t current_layer = 1; current_layer < all_layer_size_; current_layer++)
                {

                    prev_output_ = (*outputs_)[current_layer - 1];
                    parameters_from_layer_ = (*parameters_)[current_layer - 1];

                    weight_ = parameters_from_layer_(
                        af::span,
                        af::seq(0, parameters_from_layer_.dims(1) - 2)
                    );

                    bias_ = parameters_from_layer_.col(parameters_from_layer_.dims(1) - 1);

                    current_output_ = af::matmul(weight_,prev_output_) + bias_;
                    current_output_.eval();

                    switch ((*all_layer_type_)[current_layer])
                    {
                    case lantern::ffn::node::NodeType::LINEAR:
                    {
                        current_output_ = lantern::activation::Linear(current_output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::SIGMOID:
                    {
                        current_output_ = lantern::activation::Sigmoid(current_output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::RELU:
                    {
                        current_output_ = lantern::activation::ReLU(current_output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::TANH:
                    {
                        current_output_ = lantern::activation::TanH(current_output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::SWISH:
                    {
                        current_output_ = lantern::activation::Swish(current_output_);

                        break;
                    }
                    }

                    (*outputs_)[current_layer] = current_output_;
                }
            }

            /**
             * @brief Initalize the layer and FFN parameters_
             * @tparam Optimizer 
             * @param _layer 
             * @param _optimizer 
             */
            template <typename Optimizer = lantern::ffn::optimizer::AdaptiveMomentEstimation>
            inline void Initialize(
                lantern::ffn::layer::Layer &_layer,
                Optimizer &_optimizer)
            {
                auto &stack_previous_gradient_ = _optimizer.GetStackPrevGrad();
                auto &vector_velocity_ = _optimizer.GetVectorVelocity();

                auto* all_node_type_ = _layer.GetAllNodeTypeOfLayer();
                auto* parameters_ = _layer.GetParameters();
                auto* outputs_ = _layer.GetOutputs();
                auto* prev_gradient_ = _layer.GetPrevradient();


                parameters_->clear();
                prev_gradient_->clear();
                stack_previous_gradient_.clear();
                vector_velocity_.clear();

                auto *layers_ = _layer.GetAllLayerSizes();
                for (uint32_t i = 0; i < (*layers_).size() - 1; i++)
                {
                    parameters_->push_back(
                        af::randn(
                            (*layers_)[i + 1],
                            (*layers_)[i] + 1,
                            f64
                        )
                    );
                    stack_previous_gradient_.push_back(
                        af::constant(
                            0.0f,
                            (*layers_)[i + 1],
                            (*layers_)[i] + 1,
                            f64
                        )
                    );
                    vector_velocity_.push_back(
                        af::constant(
                            0.0f,
                            (*layers_)[i + 1],
                            (*layers_)[i] + 1,
                            f64
                        )
                    );
                    prev_gradient_->push_back(
                        af::constant(
                            0.0f,
                            (*layers_)[i],
                            1,
                            f64
                        )
                    );
                    outputs_->push_back(
                        af::constant(
                            0.0f,
                            (*layers_)[i],
                            1,
                            f64
                        )
                    );

                    switch ((*all_node_type_)[i])
                    {
                    case lantern::ffn::node::NodeType::SIGMOID:
                    case lantern::ffn::node::NodeType::TANH:
                    {
                        lantern::init::XavierNormInit(
                            (*layers_)[i],
                            (*layers_)[i + 1],
                            parameters_->back()
                        );
                        break;
                    }
                    case lantern::ffn::node::NodeType::RELU:
                    case lantern::ffn::node::NodeType::LINEAR:
                    case lantern::ffn::node::NodeType::SWISH:
                    {
                        lantern::init::XavierUnifInit(
                            (*layers_)[i],
                            (*layers_)[i + 1],
                            parameters_->back()
                        );
                        break;
                    }
                    }

                    // set bias_ to be 0
                    af::array &params_ = parameters_->back();
                    params_.col(params_.dims(1) - 1) = af::constant(0.0f, params_.dims(0), f64);
                }

                outputs_->push_back(
                    af::constant(
                        0.0f,
                        (*layers_).back(),
                        1,
                        f64
                    )
                );

                prev_gradient_->push_back(
                    af::array()
                );
            }
        }
    }

}
