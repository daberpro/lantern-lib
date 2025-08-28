#pragma once
#include "../pch.h"
#include "../Headers/Function.h"
#include "FFNLayer.h"
#include "FFNNode.h"
#include "FFNRegularization.h"

/**
 * @defgroup LanternBackprop Backpropagation function for lantern
 */

namespace lantern
{

    namespace ffn
    {

        namespace backprop
        {

            /**
             * @brief Backpropagate thorught model
             *
             * @tparam Optimizer
             * @tparam RegularizationFunction
             * @param _layer Layer of model
             * @param _parameters Stack of weights_ ans bias
             * @param _prev_gradient Stack of gradient_ need to compute all weights_ and bias
             * @param _outputs Stack of input_
             * @param _optimizer Optimizer for model
             * @param _batch_size Batch size
             * @ingroup LanternBackprop
             */
            template <typename Optimizer>
            inline void Backpropagate(
                lantern::ffn::layer::Layer &_layer,
                Optimizer &_optimizer,
                const uint32_t _batch_size = 1)
            {

                auto *parameters_ = _layer.GetParameters();
                auto *outputs_ = _layer.GetOutputs();
                auto *prev_gradient_ = _layer.GetPrevradient();

                double batch_size = static_cast<double>(_batch_size);
                lantern::utility::Vector<uint32_t> *all_layer_sizes = _layer.GetAllLayerSizes();
                lantern::utility::Vector<lantern::ffn::node::NodeType> *all_layer_type = _layer.GetAllNodeTypeOfLayer();
                af::array input_, output_, gradient_, gradient_weight_, gradient_bias_, all_gradient_, weights_;

                for (uint32_t current_layer = (*all_layer_sizes).size() - 1; current_layer > 0; current_layer--)
                {

                    output_ = (*outputs_)[current_layer];
                    input_ = (*outputs_)[current_layer - 1];
                    af::array& parameters_from_layer_ = (*parameters_)[current_layer - 1];
                    weights_ = parameters_from_layer_.cols(0, parameters_from_layer_.dims(1) - 2);

                    switch ((*all_layer_type)[current_layer])
                    {
                    case lantern::ffn::node::NodeType::LINEAR:
                    {
                        gradient_ = lantern::derivative::Linear(output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::SIGMOID:
                    {
                        gradient_ = lantern::derivative::Sigmoid(output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::RELU:
                    {
                         gradient_ = lantern::derivative::ReLU(output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::TANH:
                    {
                        // gradient_ = lantern::derivative::TanH(output_);

                        break;
                    }
                    case lantern::ffn::node::NodeType::SWISH:
                    {
                        gradient_ = lantern::derivative::Swish(output_);

                        break;
                    }
                    }

                    /*
                    ===========================================================================
                    Below is the visual of how this gradient_ calculate works
                    ===========================================================================
                                                                ┌───────────┐
                                                                │           │
                                                                │           │
                                                                │           │
                                                                │           │
                        f(parameters * Prev_Input) = input ──▶  │   Node    | ───▶ Output
                                                                │           │
                                                                │           │
                                                                │           │
                                                                │           │
                                                                └───────────┘
                    ===========================================================================
                    */

                    gradient_ *= (*prev_gradient_)[current_layer];
                    gradient_.eval();
                    gradient_weight_ = af::matmul(gradient_, input_.T());

                    gradient_weight_.eval();
                    gradient_bias_ = gradient_;

                    all_gradient_ = af::join(
                        1,
                        gradient_weight_,
                        gradient_bias_
                    );

                    all_gradient_ /= batch_size;
                    all_gradient_.eval();

                    uint32_t opt_index_ = current_layer - 1;
                    parameters_from_layer_ -= _optimizer.GetDelta(all_gradient_, opt_index_);
                    parameters_from_layer_.eval();

                    (*prev_gradient_)[current_layer - 1] = af::matmul(
                        parameters_from_layer_(af::span, af::seq(parameters_from_layer_.dims(1) - 1)).T(),
                        gradient_
                    );
                }
            }

        }
    }

}