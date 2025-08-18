#pragma once
#include "../pch.h"
#include "../Headers/Function.h"
#include "CNNOptimizer/CNNOptimizer.h"
#include "CNNLayer.h"
#include "CNNNode.h"

namespace lantern {


		namespace cnn {

			namespace feedforward {

				
				inline void FeedForward(
					lantern::cnn::layer::Layer& _layer
				) {

					auto all_node_type_ = _layer.GetAllNodeTypeOfLayer();
					auto all_convolve_layer_info_ = _layer.GetAllConvolveLayerInfo();
					auto all_pooling_layer_info_ = _layer.GetAllPoolingLayerInfo();
					auto all_layer_size_ = _layer.GetAllLayerSizes();
					auto pooling_modification_input_result_ = _layer.GetPoolingModificationInputResult();
					auto batch_norm_derivative_params_ = _layer.GetBatchNormDerivativeParams();
					auto batch_norm_params_ = _layer.GetBatchNormParams();

					auto weights_ = _layer.GetWeights();
					auto bias_ = _layer.GetBias();
					auto prev_gradient_ = _layer.GetPrevGradient();
					auto outputs_ = _layer.GetOutputs();
					
					// remember to clear the temp container to hold
					// data for derivative in backpropagation
					pooling_modification_input_result_->clean();
					batch_norm_derivative_params_->clean();

					uint32_t prev_gradient_index  = 0,batch_norm_index = 0,convolve_index_ = 0, pooling_index_ = 0, width_ = 0, height_ = 0, depth_ = 0, out_width_ = 0, out_height_ = 0;
					uint32_t rest_width = 0, rest_height = 0, padding_width = 0, padding_height = 0;
					af::array output, prev_outputs;


					for (uint32_t i = 0; i < all_layer_size_->size(); i++) {

						prev_outputs = (*outputs_)[i];
						
						switch ((*all_node_type_)[i]) {
						case lantern::cnn::node::NodeType::CONVOLVE: {
							lantern::cnn::layer::ConvolveLayerInfo& convolve_info_ = (*all_convolve_layer_info_)[convolve_index_];
							output = af::convolve2NN(
								prev_outputs,
								(*weights_)[convolve_index_],
								convolve_info_.stride,
								convolve_info_.padding,
								af::dim4(1, 1)
							) + (*bias_)[convolve_index_];
							output.eval();

							if (convolve_index_ + 1 < all_convolve_layer_info_->size()) {
								convolve_index_++;
							}

							prev_gradient_index++;

							break;
						}
						case lantern::cnn::node::NodeType::SWISH: {
							output = lantern::activation::Swish(prev_outputs);
							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::SIGMOID: {
							output = lantern::activation::Sigmoid(prev_outputs);
							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::RELU: {
							output = lantern::activation::ReLU(prev_outputs);
							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::LEAKY_RELU: {
							output = lantern::activation::LeakyReLU(prev_outputs);
							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::MAX_POOL: {
							lantern::cnn::layer::PoolingLayerInfo& pooling_info_ = (*all_pooling_layer_info_)[pooling_index_];
							std::pair<af::array,af::array> out_result_ = lantern::pooling::MaxPoolWithStride(
								prev_outputs,
								pooling_info_.size_h,
								pooling_info_.size_w,
								pooling_info_.stride
							);

							output = out_result_.first;
							pooling_modification_input_result_->push_back(out_result_.second);

							if (pooling_index_ + 1 < all_pooling_layer_info_->size()) {
								pooling_index_++;
							}

							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::AVG_POOL: {
							lantern::cnn::layer::PoolingLayerInfo& pooling_info_ = (*all_pooling_layer_info_)[pooling_index_];
							std::pair<af::array,af::array> out_result_ = lantern::pooling::AvgPoolWithStride(
								prev_outputs,
								pooling_info_.size_h,
								pooling_info_.size_w,
								pooling_info_.stride
							);

							output = out_result_.first;
							pooling_modification_input_result_->push_back(out_result_.second);

							if (pooling_index_ + 1 < all_pooling_layer_info_->size()) {
								pooling_index_++;
							}
							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::FLATTEN: {

							output = af::flat(prev_outputs);
							prev_gradient_index++;
							break;
						}
						case lantern::cnn::node::NodeType::BATCH_NORM : {

							af::array& batch_norm_param = (*batch_norm_params_)[batch_norm_index];
							output = lantern::normalize::BatchNorm(
								prev_outputs, 
								batch_norm_param,
								(*prev_gradient_)[prev_gradient_index],
								(*batch_norm_derivative_params_)[batch_norm_index]
							);

							batch_norm_index++;
							prev_gradient_index++;
							
						}
						}

						(*outputs_)[i + 1] = output;

					}

				}

				
				/**
				 * @brief Initialize CNN parameters
				 * @tparam Optimizer 
				 * @param _layer 
				 * @param _weights 
				 * @param _bias 
				 * @param _prev_gradient 
				 * @param _outputs 
				 * @param _optimizer 
				 * @ingroup LanternFeedForward
				 */
				template <typename Optimizer = lantern::cnn::optimizer::GradientDescent>
				inline void Initialize(
					lantern::cnn::layer::Layer& _layer,
					Optimizer& _optimizer
				) {

					auto& w_stack_previous_gradient_ = _optimizer.GetWStackPrevGrad();
					auto& w_vector_velocity_ = _optimizer.GetWVectorVelocity();
					auto& b_stack_previous_gradient_ = _optimizer.GetBStackPrevGrad();
					auto& b_vector_velocity_ = _optimizer.GetBVectorVelocity();
					auto& other_stack_previous_gradient_ = _optimizer.GetBatchNormStackPreviousGradient();
					auto& other_vector_velocity_ = _optimizer.GetBatchNormVectorVelocity();

					auto all_node_type_ = _layer.GetAllNodeTypeOfLayer();
					auto all_convolve_layer_info_ = _layer.GetAllConvolveLayerInfo();
					auto all_pooling_layer_info_ = _layer.GetAllPoolingLayerInfo();
					auto all_layer_size_ = _layer.GetAllLayerSizes();
					auto pooling_modification_input_result = _layer.GetPoolingModificationInputResult();
					auto input_size_ = _layer.GetInputSize();
					auto batch_norm_params_ = _layer.GetBatchNormParams();
					auto batch_norm_derivative_params_ = _layer.GetBatchNormDerivativeParams();
					// TODO ADD BATCH NORM 
					/*auto batch_norm_*/

					auto outputs_ = _layer.GetOutputs();
					auto weights_ = _layer.GetWeights();
					auto prev_gradient_ = _layer.GetPrevGradient();
					auto bias_ = _layer.GetBias();

					// clean all parameters and value
					weights_->clean();
					outputs_->clean();
					bias_->clean();
					prev_gradient_->clean();

					w_stack_previous_gradient_.clean();
					w_vector_velocity_.clean();
					b_stack_previous_gradient_.clean();
					b_vector_velocity_.clean();
					other_stack_previous_gradient_.clean();
					other_vector_velocity_.clean();

					pooling_modification_input_result->clean();
					batch_norm_params_->clean();
					batch_norm_derivative_params_->clean();

					// the first outputs is input image
					outputs_->push_back(
						af::constant(
							0.0f,
							(*input_size_)[0],
							(*input_size_)[1],
							(*input_size_)[2],
							f64
						)
					);

					uint32_t convolve_index_ = 0, pooling_index_ = 0, width_ = 0, height_ = 0, depth_ = 0, out_width_ = 0, out_height_ = 0;
					uint32_t rest_width = 0, rest_height = 0, total_removed_rows_ = 0, total_removed_cols_ = 0;

					for (uint32_t i = 0; i < all_layer_size_->size(); i++) {

						af::array& prev_outputs_ = outputs_->back();
						width_ = prev_outputs_.dims(0);
						height_ = prev_outputs_.dims(1);
						depth_ = prev_outputs_.dims(2);

						if (width_ == 0 || height_ == 0 || depth_ == 0) {
							throw std::runtime_error(std::format("Width, Height, or Depth of prev output cannot be zero at Layer [{}]\n", outputs_->size()));
						}

						// check the type of node first
						switch ((*all_node_type_)[i]) {
						case lantern::cnn::node::NodeType::CONVOLVE: {

							// get convolve info and pooling info from layer
							lantern::cnn::layer::ConvolveLayerInfo& convolve_info_ = (*all_convolve_layer_info_)[convolve_index_];

							// prevent convolve with small input
							if(width_ < convolve_info_.kernel_size || height_ < convolve_info_.kernel_size){
								throw std::runtime_error(std::format("Width or Height input cannot smaller than kernel size at Layer [{}]\n", outputs_->size()));
							}

							out_width_ = (width_ - convolve_info_.kernel_size + 2 * convolve_info_.padding[0]) / convolve_info_.stride[0] + 1;
							out_height_ = (height_ - convolve_info_.kernel_size + 2 * convolve_info_.padding[1]) / convolve_info_.stride[1] + 1;

							weights_->push_back(
								af::randn(
									convolve_info_.kernel_size, // kernel width
									convolve_info_.kernel_size, // kernel height
									convolve_info_.kernel_depth, // kernel depth
									(*all_layer_size_)[i], // total kernel
									f64
								)
							);

							w_vector_velocity_.push_back(
								af::constant(
									0.0f,
									convolve_info_.kernel_size, // kernel width
									convolve_info_.kernel_size, // kernel height
									convolve_info_.kernel_depth, // kernel depth
									(*all_layer_size_)[i], // total kernel
									f64
								)
							);

							w_stack_previous_gradient_.push_back(
								af::constant(
									0.0f,
									convolve_info_.kernel_size, // kernel width
									convolve_info_.kernel_size, // kernel height
									convolve_info_.kernel_depth, // kernel depth
									(*all_layer_size_)[i], // total kernel
									f64
								)
							);

							prev_gradient_->push_back(
								af::constant(
									0.0f,
									out_width_,
									out_height_,
									(*all_layer_size_)[i],
									f64
								)
							);

							outputs_->push_back(
								af::constant(
									0.0f,
									out_width_,
									out_height_,
									(*all_layer_size_)[i],
									f64
								)
							);

							bias_->push_back(
								af::constant(
									0.0f,
									1,
									1,
									(*all_layer_size_)[i],
									f64
								)
							);

							b_stack_previous_gradient_.push_back(
								af::constant(
									0.0f,
									1,
									1,
									(*all_layer_size_)[i],
									f64
								)
							);

							b_vector_velocity_.push_back(
								af::constant(
									0.0f,
									1,
									1,
									(*all_layer_size_)[i],
									f64
								)
							);

							// must check if the convolve index not greather than all_convolve_layer_info_->size()
							// cause it will make the out_width_ and out_height_ divide by zero
							if (convolve_index_ + 1 < all_convolve_layer_info_->size()) {
								convolve_index_++;
							}

							break;
						}
						case lantern::cnn::node::NodeType::AVG_POOL: 
						case lantern::cnn::node::NodeType::MAX_POOL: {

							lantern::cnn::layer::PoolingLayerInfo& pooling_info_ = (*all_pooling_layer_info_)[pooling_index_];

							if(width_ < pooling_info_.size_w || height_ < pooling_info_.size_h){
								throw std::runtime_error(std::format("Width or Height input cannot smaller than pooling size at Layer [{}]\n", outputs_->size()));
							}
							rest_width = width_ % pooling_info_.size_w; // Get valid width of pooling 
							rest_height = height_ % pooling_info_.size_h; // Get valid height of 

							// Output for Pooling
							outputs_->push_back(
								af::constant(
									0.0f,
									/*
									* Out width -> ((prev_out_width / stride_w) - (prev_out_width / stride_w)  % stride_w) / pool_w
         							* Out height -> ((prev_out_height / stride_h) - (prev_out_height / stride_h) % stride_h) / poll_h
									*/
									((width_ / pooling_info_.stride[0]) - rest_width) / pooling_info_.size_w,
									((height_ / pooling_info_.stride[1]) - rest_height) / pooling_info_.size_h,
									depth_,
									f64
								)
							);

							prev_gradient_->push_back(
								af::constant(
									0.0f,
									outputs_->back().dims(),
									f64
								)
							);

							// must check if the pooling index not greather than all_pooling_layer_info->size()
							// cause it will make the out_width_ and out_height_ divide by zero
							if (pooling_index_ + 1 < all_pooling_layer_info_->size()) {
								pooling_index_++;
							}

							break;

						}
						case lantern::cnn::node::NodeType::SWISH: {

							outputs_->push_back(
								af::constant(
									0.0f,
									width_,
									height_,
									depth_,
									f64
								)
							);

							prev_gradient_->push_back(
								af::constant(
									0.0f,
									outputs_->back().dims(),
									f64
								)
							);

							break;

						}
						case lantern::cnn::node::NodeType::LEAKY_RELU:
						case lantern::cnn::node::NodeType::RELU: {

							outputs_->push_back(
								af::constant(
									0.0f,
									width_,
									height_,
									depth_,
									f64
								)
							);

							prev_gradient_->push_back(
								af::constant(
									0.0f,
									outputs_->back().dims(),
									f64
								)
							);

							break;
						}
						case lantern::cnn::node::NodeType::FLATTEN: {
							
							outputs_->push_back(
								af::flat(outputs_->back())
							);
							
							prev_gradient_->push_back(
								af::constant(
									0.0f,
									outputs_->back().dims(),
									f64
								)
							);

							break;
						}
						case lantern::cnn::node::NodeType::BATCH_NORM:{

							outputs_->push_back(
								af::constant(
									0.0f,
									outputs_->back().dims(),
									f64
								)
							);

							other_stack_previous_gradient_.push_back(
								af::constant(
									0.0f,
									1,
									2,
									outputs_->back().dims(2),
									f64
								)
							);

							other_vector_velocity_.push_back(
								af::constant(
									0.0f,
									1,
									2,
									outputs_->back().dims(2),
									f64
								)
							);

							prev_gradient_->push_back(
								af::constant(
									0.0f,
									1,
									1,
									outputs_->back().dims(2),
									f64
								)
							);

							batch_norm_params_->push_back(
								af::randn(
									1,
									2,
									outputs_->back().dims(2),
									f64
								)
							);

							batch_norm_derivative_params_->push_back(
								af::constant(
									0.0f,
									1,
									2,
									outputs_->back().dims(2),
									f64
								)
							);

							break;
						}
						}

					}

					// because pooling modification input result is juat a temp container
					// to hold the modification input from pooling activation function with stride
					// we just specify the size of it
					pooling_modification_input_result->resizeCapacity(pooling_index_);


				}

			}

		}


}