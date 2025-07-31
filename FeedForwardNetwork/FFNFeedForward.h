#pragma once
#include "../Headers/Vector.h"
#include "../pch.h"
#include "../Headers/Function.h"
#include "../Headers/Initialize.h"
#include "FFNNode.h"
#include "FFNLayer.h"
#include "FFNOptimizer/FFNOptimizer.h"

namespace lantern
{

    namespace ffn
    {

        namespace feedforward
        {

            /**
             * @brief Feed Forward
             *
             * @param _layer
             * @param _outputs
             * @param _parameters
             */
            void FeedForward(
                lantern::ffn::layer::Layer &_layer,
                lantern::utility::Vector<af::array> &_outputs,
                lantern::utility::Vector<af::array> &_parameters)
            {

                af::array parameters, prev_output, current_output, weight, bias;
                lantern::utility::Vector<uint32_t> *all_layer_sizes = _layer.GetAllLayerSizes();
                lantern::utility::Vector<lantern::ffn::node::NodeType> *all_layer_type = _layer.GetAllNodeTypeOfLayer();

                for (uint32_t current_layer = 0; current_layer < (*all_layer_sizes).size() - 1; current_layer++)
                {

                    prev_output = _outputs[current_layer];
                    parameters = _parameters[current_layer];

                    weight = parameters(
                        af::span,
                        af::seq(0, parameters.dims(1) - 2)
                    );

                    bias = parameters.col(parameters.dims(1) - 1);

                    current_output = af::matmul(weight,prev_output) + bias;
                    current_output.eval();

                    // adding 1 to skip input layer
                    switch ((*all_layer_type)[current_layer + 1])
                    {
                    case lantern::ffn::node::NodeType::LINEAR:
                    {
                        current_output = lantern::activation::Linear(current_output);

                        break;
                    }
                    case lantern::ffn::node::NodeType::SIGMOID:
                    {
                        current_output = lantern::activation::Sigmoid(current_output);

                        break;
                    }
                    case lantern::ffn::node::NodeType::RELU:
                    {
                        current_output = lantern::activation::ReLU(current_output);

                        break;
                    }
                    case lantern::ffn::node::NodeType::TANH:
                    {
                        current_output = lantern::activation::TanH(current_output);

                        break;
                    }
                    case lantern::ffn::node::NodeType::SWISH:
                    {
                        current_output = lantern::activation::Swish(current_output);

                        break;
                    }
                    }

                    // adding 1 to skip input layer
                    _outputs[current_layer + 1] = current_output;
                }
            }

            template <typename Optimizer = lantern::ffn::optimizer::AdaptiveMomentEstimation>
            void Initialize(
                lantern::ffn::layer::Layer &_layer,
                lantern::utility::Vector<af::array> &_parameters,
                lantern::utility::Vector<af::array> &_prev_gradient,
                lantern::utility::Vector<af::array> &_outputs,
                Optimizer &_optimizer)
            {
                auto &stack_previous_gradient = _optimizer.GetStackPrevGrad();
                auto &vector_velocity = _optimizer.GetVectorVelocity();
                auto *all_node_type = _layer.GetAllNodeTypeOfLayer();

                _parameters.clear();
                _prev_gradient.clear();
                stack_previous_gradient.clear();
                vector_velocity.clear();

                auto *layers = _layer.GetAllLayerSizes();
                for (uint32_t i = 0; i < (*layers).size() - 1; i++)
                {
                    _parameters.push_back(
                        af::randn(
                            (*layers)[i + 1],
                            (*layers)[i] + 1,
                            f64));
                    stack_previous_gradient.push_back(
                        af::constant(
                            0.0f,
                            (*layers)[i + 1],
                            (*layers)[i] + 1,
                            f64));
                    vector_velocity.push_back(
                        af::constant(
                            0.0f,
                            (*layers)[i + 1],
                            (*layers)[i] + 1,
                            f64));
                    _prev_gradient.push_back(
                        af::constant(
                            0.0f,
                            (*layers)[i],
                            1,
                            f64));
                    _outputs.push_back(
                        af::constant(
                            0.0f,
                            (*layers)[i],
                            1,
                            f64));

                    switch ((*all_node_type)[i])
                    {
                    case lantern::ffn::node::NodeType::SIGMOID:
                    case lantern::ffn::node::NodeType::TANH:
                    {
                        lantern::init::XavierNormInit(
                            (*layers)[i],
                            (*layers)[i + 1],
                            _parameters.back());
                        break;
                    }
                    case lantern::ffn::node::NodeType::RELU:
                    case lantern::ffn::node::NodeType::LINEAR:
                    case lantern::ffn::node::NodeType::SWISH:
                    {
                        lantern::init::XavierUnifInit(
                            (*layers)[i],
                            (*layers)[i + 1],
                            _parameters.back());
                        break;
                    }
                    }

                    // set bias to be 0
                    af::array &params = _parameters.back();
                    params.col(params.dims(1) - 1) = af::constant(0.0f, params.dims(0), f64);
                }

                _outputs.push_back(
                    af::constant(
                        0.0f,
                        (*layers).back(),
                        1,
                        f64
                    )
                );
            }
        }
    }

}
