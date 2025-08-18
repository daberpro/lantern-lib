#pragma once
#include "../pch.h"
#include "../ConvolutionalNeuralNetwork/CNNNode.h"

#define LANTERN_GET_FUNC_NAME(Func) #Func

/**
 * @defgroup LanternFunction All definition of functions in lantern
 */

namespace lantern{

    namespace utility {

        /**
         * @brief get max value of two params, this is use to replace default MIN_MAX macro
         * @tparam T 
         * @param a 
         * @param b 
         * @return 
         * @ingroup LanternFunction
         */
        template <typename T>
        inline T max(const T& a, const T& b){
            return (a < b? b : a);
        }

    }
    
    namespace activation {

        /**
         * @brief Sigmoid activation function
         * @param value 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array Sigmoid(const af::array& value){
            return 1/(1 + af::exp(-value));
        }

        /**
         * @brief Linear activation function
         * @param value
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array Linear(const af::array& value){
            return value;
        }

        /**
         * @brief ReLU activation function
         * @param value
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array ReLU(const af::array& value){
            return af::max(0,value);
        }

        /**
         * @brief LeakyReLU activation function
         * @param value
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array LeakyReLU(const af::array& value){
            return af::max(value * 0.1, value);
        }

        /**
         * @brief TanH activation function
         * @param value
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array TanH(const af::array& value){
            af::array exp_ = af::exp(value);
            af::array nexp_ = af::exp(-value);
            return (exp_ - nexp_)/(exp_ + nexp_);
        }

        /**
         * @brief Swish activation function
         * @param value
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array Swish(const af::array& value){
            return value * Sigmoid(value);
        }

        /**
         * @brief Get Activation Map, this is to get activation function from string
         * @return std::unordered_map<std::string, std::function<af::array(const af::array&)>>&
         * @ingroup LanternFunction
         */
        inline std::unordered_map<std::string, std::function<af::array(const af::array&)>>& GetActivationMap() {
            static std::unordered_map<std::string, std::function<af::array(const af::array&)>> map = {
                {"lantern::activation::Sigmoid", Sigmoid},
                {"lantern::activation::Linear", Linear},
                {"lantern::activation::ReLU", ReLU},
                {"lantern::activation::LeakyReLU", LeakyReLU},
                {"lantern::activation::TanH", TanH},
                {"lantern::activation::Swish", Swish}
            };
            return map;
        }

    }

    namespace normalize {

        /**
         * =========================================================================================
         * @brief Applies batch normalization to the input value using beta and gamma parameters.
         *  This function computes the mean and variance of the input, normalizes the input,
         *  and applies the beta and gamma parameters to produce the output.
         * @param _value  The input ArrayFire array to be normalized.
         * @param _beta   The beta parameter for batch normalization.
         * @param _gamma  The gamma parameter for batch normalization.
         * @return A tuple containing:
         *         - The normalized output array.
         *         - The gradient of the input value with respect to the output.
         *         - The gradient of the beta and gamma parameters.
         * @ingroup LanternFunction
         * * =========================================================================================
         */
        inline af::array BatchNorm(const af::array& _value, const af::array& _beta_gamma, af::array& _dvalue, af::array& _dbeta_gamma) {
            
            af::array mean = af::mean(af::mean(_value,0),1);
            af::array var = af::sum(af::sum(af::pow(_value - mean,2),0),1) / (_value.dims(0) * _value.dims(1));
            
            af::array norm = (_value - mean) / af::sqrt(var + 1e-12);
            af::array output = _beta_gamma(0,0) * norm + _beta_gamma(0,1);

            // calculate gradient
            af::array doutput_dnorm = _beta_gamma(0,0);
            af::array dnorm_dvalue = 1 / af::sqrt(var + 1e-12);
            af::array dnorm_dmean = -1 / af::sqrt(var + 1e-12);
            af::array dnorm_dvar = -0.5 * (_value - mean) / af::pow(var + 1e-12, 1.5);
            af::array dvalue = doutput_dnorm * dnorm_dvalue + 
                               af::sum(doutput_dnorm * dnorm_dmean) / _value.dims(0) + 
                               af::sum(doutput_dnorm * dnorm_dvar * 2 * (_value - mean)) / _value.dims(0);
            af::array dgamma = af::sum(doutput_dnorm * norm) / _value.dims(0);
            af::array dbeta = af::sum(doutput_dnorm) / _value.dims(0);
            af::array gradient = af::join(1, dgamma, dbeta); // Concatenate the gradients for value, gamma, and beta
            
            return output; 
        }

    }

    namespace pooling {
        

        /**
         * =========================================================================================
         * @brief Performs max pooling on a 3D/2D ArrayFire input using specified pooling dimensions.
         *
         * The function reshapes and reorders the input to create pooling blocks,
         * applies max operations across spatial axes, and finally reshapes
         * the result to match the reduced dimensions.
         *
         * @param _input   The original ArrayFire array (typically 3D: width × height × depth).
         * @param _pool_h  Pooling window height.
         * @param _pool_w  Pooling window width.
         * @return         A pooled ArrayFire array transposed after max operations,
         *                 with dimensions reduced by pool_h and pool_w.
         * @ingroup LanternFunction
         * =========================================================================================
         */
         inline af::array MaxPool(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w) {

            int32_t dims_0_, dims_1_, dims_2_;
            af::array res_ = af::reorder(_input, 0, 2, 1),sliced_derivative_;
            res_ = af::moddims(res_, res_.dims(0), res_.dims(1) * res_.dims(2), 1);
            dims_0_ = res_.dims(0);
            dims_1_ = res_.dims(1);
            dims_2_ = res_.dims(2);
            res_ = af::moddims(res_, _pool_h, res_.dims(1) / _pool_h, res_.dims(0));
            res_ = af::reorder(res_,0,2,1);
            res_ = af::moddims(res_,_pool_h,_pool_w,(res_.dims(1) * res_.dims(0)) / (_pool_w * _pool_h) * res_.dims(2));
            res_ = af::max(res_,1);
            res_ = af::max(res_,0);
            res_ = af::reorder(res_, 2, 1, 0);
            res_ = af::moddims(res_, dims_0_ / _pool_h, dims_1_ / _pool_w, 1);
            res_ = af::moddims(res_, _input.dims(0) / _pool_h, _input.dims(1) / _pool_w, _input.dims(2));
            
            return res_.T();
        }

        /**
        * =========================================================================================
        * @brief Performs avg pooling on a 3D/2D ArrayFire input using specified pooling dimensions.
        *
        * The function reshapes and reorders the input to create pooling blocks,
        * applies avg operations across spatial axes, and finally reshapes
        * the result to match the reduced dimensions.
        *
        * @param _input   The original ArrayFire array (typically 3D: width × height × depth).
        * @param _pool_h  Pooling window height.
        * @param _pool_w  Pooling window width.
        * @return         A pooled ArrayFire array transposed after max operations,
        *                 with dimensions reduced by pool_h and pool_w.
        * @ingroup LanternFunction
        * =========================================================================================
        */
        inline af::array AvgPool(const af::array& input, const uint32_t& pool_h,const uint32_t& pool_w) {

            int32_t dims_0, dims_1, dims_2;
            af::array res = af::reorder(input, 0, 2, 1);
            res = af::moddims(res, res.dims(0), res.dims(1) * res.dims(2), 1);
            dims_0 = res.dims(0);
            dims_1 = res.dims(1);
            dims_2 = res.dims(2);
            res = af::moddims(res, pool_h, res.dims(1) / pool_h, res.dims(0));
            res = af::reorder(res, 0, 2, 1);
            res = af::moddims(res, pool_h, pool_w, (res.dims(1) * res.dims(0)) / (pool_w * pool_h) * res.dims(2));
            res = af::sum(res, 1);
            res = af::sum(res, 0) / (pool_h * pool_w);
            res.eval();
            res = af::reorder(res, 2, 1, 0);
            res = af::moddims(res, dims_0 / pool_h, dims_1 / pool_w, 1);
            res = af::moddims(res, input.dims(0) / pool_h, input.dims(1) / pool_w, input.dims(2));

            return res.T();
        }

        /**
         * @brief Pooling with stride, the pooling types must be defined
         * @tparam PoolType 
         * @param _input 
         * @param _pool_h 
         * @param _pool_w 
         * @param _stride 
         * @return std::pair<af::array,af::array>
         * @ingroup LanternFunction
         */
        template <lantern::cnn::node::NodeType PoolType>
        inline std::pair<af::array,af::array> PoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const af::dim4& _stride){
             /**
             * Get pooling size after stride apply
             */
            af::array res_ = _input;
            res_ = res_(
                af::seq(0,res_.dims(0) - 1,_stride[1]), // remember stride[1] is height
                af::seq(0,res_.dims(1) - 1,_stride[0]), // stride[0] is width
                af::span
            );
            
            /**
             * get the rest of width and height which is not valid pooling
             * and get size of them to remove it from res_ array
             */
            uint32_t width_ = res_.dims(0);
            uint32_t height_ = res_.dims(1);
            uint32_t rest_width_ = width_ % _pool_w; // Get invalid width of pooling 
            uint32_t rest_height_ = height_ % _pool_h; // Get invalid height of pooling 

            /**
             * create a mask rows and columns
             * to get all rows and cols index to remove
             */
            af::array mask_rows_ = af::constant(1.0,res_.dims(0), b8);
            af::array mask_cols_ = af::constant(1.0,res_.dims(1), b8);
            uint32_t total_remove_cols_ = (rest_width_ == 0) ? 0 : (_pool_w - rest_width_); // Get the required size to remove for valid pooling from rest width
            uint32_t total_remove_rows_ = (rest_height_ == 0) ? 0 : (_pool_h - rest_height_);  // Get the required size to remove for valid pooling from rest height
            
            /**
             * Check if total remove rows or cols are zero
             * we just skip to remove it, and without conditional block 
             * the last cols or rows will remove
             */
            if(total_remove_rows_ > 0){
                mask_rows_(af::seq(mask_rows_.dims(0) - total_remove_rows_, mask_rows_.dims(0) - 1, 1)) = 0;
            }
            if(total_remove_cols_ > 0){
                mask_cols_(af::seq(mask_cols_.dims(0) - total_remove_cols_, mask_cols_.dims(0) - 1, 1)) = 0;
            }

            /**
             * Then we get the index of rows and columns will pass to MaxPool
             */
            af::array valid_rows = af::where(mask_rows_); 
            af::array valid_cols = af::where(mask_cols_); 
            af::array valid_input_ = res_(valid_rows,valid_cols);

            af::array result;

            switch (PoolType)
            {
                case lantern::cnn::node::NodeType::MAX_POOL:{
                    result = MaxPool(
                        valid_input_,
                        _pool_h,
                        _pool_w
                    );
                    break;
                }
                case lantern::cnn::node::NodeType::AVG_POOL:{
                    result = AvgPool(
                        valid_input_,
                        _pool_h,
                        _pool_w
                    );
                    break;
                }
                default:{
                    throw std::runtime_error("lantern::pooling::PoolWithStride<PoolingType>() error unknown pooling type!");
                }
            }

            return {
                result,
                valid_input_ // this is use for backpropagation of Pooling
            };
        }
        
        /**
         * =========================================================================================
         * @brief Applies max pooling with custom stride to the input ArrayFire array.
         *
         * This function subsamples the input array using stride parameters, pads the result
         * to ensure pooling dimensions fit evenly, and returns both the pooled result and
         * the padded array used in the computation.
         *
         * @param _input    The input ArrayFire array to be pooled.
         * @param _pool_h   Pooling window height.
         * @param _pool_w   Pooling window width.
         * @param _stride   A dim4 object specifying stride for width and height (stride[0], stride[1]).
         * @return          A std::pair where:
         *                  - first  => Result of max pooling after stride and padding.
         *                  - second => Padded array used for pooling (post-stride).
         * @ingroup LanternFunction
         * =========================================================================================
         */
        inline std::pair<af::array,af::array> MaxPoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const af::dim4& _stride){
            return PoolWithStride<lantern::cnn::node::NodeType::MAX_POOL>(_input,_pool_h,_pool_w,_stride);
        }

        /**
         * =========================================================================================
         * @brief Applies avg pooling with custom stride to the input ArrayFire array.
         *
         * This function subsamples the input array using stride parameters, pads the result
         * to ensure pooling dimensions fit evenly, and returns both the pooled result and
         * the padded array used in the computation.
         *
         * @param _input    The input ArrayFire array to be pooled.
         * @param _pool_h   Pooling window height.
         * @param _pool_w   Pooling window width.
         * @param _stride   A dim4 object specifying stride for width and height (stride[0], stride[1]).
         * @return          A std::pair where:
         *                  - first  => Result of max pooling after stride and padding.
         *                  - second => Padded array used for pooling (post-stride).
         * @ingroup LanternFunction
         * =========================================================================================
         */
        inline std::pair<af::array,af::array> AvgPoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const af::dim4& _stride) {
            return PoolWithStride<lantern::cnn::node::NodeType::AVG_POOL>(_input,_pool_h,_pool_w,_stride);
        }

        /**
         * @brief Get pooling Map, this is to get pooling function from string
         * @return std::unordered_map<std::string, std::function<std::pair<af::array,af::array>(const af::array&, uint32_t const&, uint32_t const&, const af::dim4&)>>&
         */
        inline std::unordered_map<std::string, std::function<std::pair<af::array,af::array>(const af::array&, uint32_t const&, uint32_t const&, const af::dim4&)>>& GetPoolingMaps() {
            static std::unordered_map<std::string, std::function<std::pair<af::array,af::array>(const af::array&, uint32_t const&, uint32_t const&, const af::dim4&)>> map = {
                {"lantern::pooling::MaxPoolWithStride", MaxPoolWithStride},
                {"lantern::pooling::AvgPoolWithStride",AvgPoolWithStride}
            };
            return map;
        }

    }


    namespace probability {

        /**
         * @brief Softmat probability function
         * @param value 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array SoftMax(const af::array& value){
            af::array exp_ = af::exp(value);
            af::array sum_exp_ = af::sum(af::exp(value));
            return exp_/sum_exp_;
        }

        /**
         * @brief Get probability Map, this is to get probability function from string
         * @return std::unordered_map<std::string, std::function<af::array(const af::array&)>>&
         * @ingroup LanternFunction
         */
        inline std::unordered_map<std::string, std::function<af::array(const af::array&)>>& GetProbabilityMaps() {
            static std::unordered_map<std::string, std::function<af::array(const af::array&)>> map = {
                {"lantern::probability::SoftMax", SoftMax}
            };
            return map;
        }

    }

    namespace loss {

        /**
         * @brief Sum squared resiudal loss function
         * @param output 
         * @param target 
         * @return double
         * @ingroup LanternFunction
         */
        inline double SumSquareResidual(const af::array& output, const af::array& target){
            return af::pow(target - output,2).scalar<double>();
        }

        /**
         * @brief Cross Entropy loss function
         * @param output
         * @param target
         * @return double
         * @ingroup LanternFunction
         */
        inline double CrossEntropy(const af::array& output, const af::array& target){
            return af::sum(-(target * af::log(output + 1e-012))).scalar<double>();
        }

        /**
         * @brief Binary Cross Entropy
         * @param output 
         * @param target 
         * @return double
         */
        inline double BinaryCrossEntropy(const af::array& output, const af::array& target) {
            af::array true_prob = target * af::log(output + 1e-012);
            af::array false_prob = (1 - target) * af::log(1 - output + 1e-012);
            return af::sum(-(true_prob + false_prob)).scalar<double>();
        }

        /**
         * @brief Get  Map, this is to get activation function from string
         * @return std::unordered_map<std::string, std::function<double(const af::array&, const af::array&)>>&
         * @ingroup LanternFunction
         */
        inline std::unordered_map<std::string, std::function<double(const af::array&, const af::array&)>>& GetLossMaps() {
            static std::unordered_map<std::string, std::function<double(const af::array&, const af::array&)>> map = {
                {"lantern::loss::SumSquareResidual",SumSquareResidual},
                {"lantern::loss::CrossEntropy", CrossEntropy},
                {"lantern::loss::BinaryCrossEntropy", BinaryCrossEntropy}
            };
            return map;
        }

    }

    namespace derivative {
        
        
        /**
         * @brief Derivative of MaxPool
         * @param _input 
         * @param _pool_h 
         * @param _pool_w 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array MaxPool(const af::array& _input, const uint32_t& _pool_h, const uint32_t& _pool_w){
           
            af::array res_ = af::reorder(_input, 0, 2, 1),sliced_derivative_;
            res_ = af::moddims(res_, res_.dims(0), res_.dims(1) * res_.dims(2), 1);
            res_ = af::moddims(res_, _pool_h, res_.dims(1) / _pool_h, res_.dims(0));
            res_ = af::reorder(res_,0,2,1);
            res_ = af::moddims(res_,_pool_h,_pool_w,(res_.dims(1) * res_.dims(0)) / (_pool_w * _pool_h) * res_.dims(2));
            sliced_derivative_ = res_;
            res_ = af::max(res_,1);
            res_ = af::max(res_,0);
            res_ = af::reorder(res_, 2, 1, 0);
            sliced_derivative_ = (sliced_derivative_ == af::tile(af::reorder(res_, 2, 1, 0), sliced_derivative_.dims(0), sliced_derivative_.dims(1))).as(f64);
            sliced_derivative_ = af::moddims(sliced_derivative_,sliced_derivative_.dims(0),_input.dims(1),(_input.dims(0) / _pool_h) * _input.dims(2));
            sliced_derivative_ = af::moddims(sliced_derivative_.T(),_input.dims(0),_input.dims(1),_input.dims(2));
            return sliced_derivative_;

        }

        /**
         * @brief Derivative of Avg Pool 
         * @param _input 
         * @param _pool_h 
         * @param _pool_w 
         * @return 
         * @ingroup LanternFunction
         */
        inline af::array AvgPool(const af::array& _input, const uint32_t& _pool_h, const uint32_t& _pool_w){
            
            return af::constant(
                1.0f/static_cast<double>(_pool_h * _pool_w),
                _input.dims(),
                f64
            );

        }

        /**
         * @brief Derivative of pooling with stride, you must specify pool type
         * @tparam PoolType 
         * @param _input 
         * @param _modify_input 
         * @param _pool_h 
         * @param _pool_w 
         * @param _stride 
         * @param _prev_gradient 
         * @return af::array
         * @ingroup LanternFunction
         */
        template <lantern::cnn::node::NodeType PoolType>
        inline af::array PoolWithStride(const af::array& _input, const af::array& _modify_input, const uint32_t& _pool_h, const uint32_t& _pool_w, const af::dim4& _stride, const af::array& _prev_gradient){
            /**
             * Create a temporary output variabel with the size same as the input
             * in feedforward
             */
            af::array temp_out = af::constant(0.0, _input.dims(),f64);
            af::array shaping_derivative_ = af::constant(0.0, _input.dims(),f64), prev_gradient_;

            // do derivative of pool
            af::array res_; 

            switch (PoolType)
            {
                case lantern::cnn::node::NodeType::MAX_POOL:{
                    res_ = MaxPool(
                        _modify_input,
                        _pool_h,
                        _pool_w
                    );
                    break;
                }
                case lantern::cnn::node::NodeType::AVG_POOL:{
                    res_ = AvgPool(
                        _modify_input,
                        _pool_h,
                        _pool_w
                    );
                    break;
                }
                default:{
                    throw std::runtime_error("lantern::derivative::PoolWithStride<PoolingType>() error unknown pooling type!");
                }
            }

            prev_gradient_ = af::constant(
                0.0f,
                 _prev_gradient.dims(0) * _pool_h, 
                 _prev_gradient.dims(1) * _pool_w, 
                 _prev_gradient.dims(2),
                 f64
            );

            prev_gradient_(
                af::seq(0,prev_gradient_.dims(0) - 1,_pool_h),
                af::seq(0,prev_gradient_.dims(1) - 1,_pool_w),
                af::span
            ) = _prev_gradient;

            /**
             * Get the dimension of preprocess input in feedforward to know 
             * what is the dimension after stride apply
             */
            af::dim4 out_dims = temp_out(
                af::seq(0,temp_out.dims(0) - 1,_stride[1]), // remember stride[1] is height
                af::seq(0,temp_out.dims(1) - 1,_stride[0]), // stride[0] is width
                af::span
            ).dims();

            /**
             * Get the removing column size to re-apply 
             * and merge them into result of maxpooling derivative
             */
            uint32_t width_ = out_dims[1];
            uint32_t height_ = out_dims[0];
            uint32_t rest_width_ = width_ % _pool_w; // Get rest of the input width as removed 
            uint32_t rest_height_ = height_ % _pool_h; // Get rest of the input height as removed

            res_ = af::join(
                1,
                res_,
                af::constant(
                    0.0,
                    res_.dims(0),
                    rest_width_,
                    res_.dims(2),
                    res_.dims(3),
                    f64
                )
            );

            res_ = af::join(
                0,
                res_,
                af::constant(
                    0.0,
                    rest_height_,
                    res_.dims(1),
                    res_.dims(2),
                    res_.dims(3),
                    f64
                )
            );

            prev_gradient_ = af::join(
                1,
                prev_gradient_,
                af::constant(
                    0.0,
                    prev_gradient_.dims(0),
                    rest_width_,
                    prev_gradient_.dims(2),
                    prev_gradient_.dims(3),
                    f64
                )
            );

            prev_gradient_ = af::join(
                0,
                prev_gradient_,
                af::constant(
                    0.0,
                    rest_height_,
                    prev_gradient_.dims(1),
                    prev_gradient_.dims(2),
                    prev_gradient_.dims(3),
                    f64
                )
            );

            /**
             * Then get all the row and columns of temporary output
             * and replace them with actual derivative of maxpool
             */
            temp_out(
                af::seq(0,temp_out.dims(0) - 1,_stride[1]),
                af::seq(0,temp_out.dims(1) - 1,_stride[0]),
                af::span,
                af::span
            ) = res_;

            shaping_derivative_(
                af::seq(0,shaping_derivative_.dims(0) - 1,_stride[1]),
                af::seq(0,shaping_derivative_.dims(1) - 1,_stride[0]),
                af::span,
                af::span
            ) = prev_gradient_;

            temp_out *= shaping_derivative_;
            temp_out.eval();

            return temp_out.T();
        }

        /**
         * @brief Derivative of MaxPool with stride
         * @param _input 
         * @param _modify_input 
         * @param _pool_h 
         * @param _pool_w 
         * @param _stride 
         * @param _prev_gradient 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array MaxPoolWithStride(const af::array& _input, const af::array& _modify_input, const uint32_t& _pool_h, const uint32_t& _pool_w, const af::dim4& _stride, const af::array& _prev_gradient){
            return PoolWithStride<lantern::cnn::node::NodeType::MAX_POOL>(_input,_modify_input,_pool_h,_pool_w,_stride,_prev_gradient);
        }

        /**
         * @brief Derivative of AvgPool with stride
         * @param _input
         * @param _modify_input
         * @param _pool_h
         * @param _pool_w
         * @param _stride
         * @param _prev_gradient
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array AvgPoolWithStride(const af::array& _input, const af::array& _modify_input, const uint32_t& _pool_h, const uint32_t& _pool_w, const af::dim4& _stride, const af::array& _prev_gradient){
            return PoolWithStride<lantern::cnn::node::NodeType::AVG_POOL>(_input,_modify_input,_pool_h,_pool_w,_stride,_prev_gradient);
        }

        /**
         * @brief Derivative of sigmoid
         * @param value 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array Sigmoid(const af::array& value){
            return value * (1 - value);
        }

        /**
         * @brief Derivative of swish
         * @param value 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array Swish(const af::array& value){
            return activation::Sigmoid(value) +  Sigmoid(value) * value;
        }

        /**
         * @brief Derivative of linear
         * @param value 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array Linear(const af::array& value){
            return af::constant(1.0f,value.dims(0),value.dims(1),f64);
        }

        /**
         * @brief Derivative of SumSquareResidual
         * @param output 
         * @param target 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array SumSquareResidual(const af::array& output, const af::array& target){
            return -2 * (target - output);
        }

        /**
         * @brief Derivative of CrossEntropy with softmax out function
         * @param output 
         * @param target 
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array CrossEntropy(const af::array& output, const af::array& target){
            return output - target;
        }

        /**
         * @brief Derivative of BinaryCrossEntropy with Sigmoid out function
         * @param output
         * @param target
         * @return af::array
         * @ingroup LanternFunction
         */
        inline af::array BinaryCrossEntropy(const af::array& output, const af::array& target) {
            return output - target;
        }

        /**
         * @brief Get pool Map, this is to get pool function from string
         * @return std::unordered_map<std::string, std::function<af::array(const af::array&, const af::array&, const uint32_t&, const uint32_t&, const af::dim4&, const af::array&)>>&
         * @ingroup LanternFunction
         */
        inline std::unordered_map<std::string, std::function<af::array(const af::array&, const af::array&, const uint32_t&, const uint32_t&, const af::dim4&, const af::array&)>>& GetDerivativePoolMaps() {
            static std::unordered_map<std::string, std::function<af::array(const af::array&, const af::array&, const uint32_t&, const uint32_t&, const af::dim4&, const af::array&)>> map = {
                {"lantern::derivative::MaxPoolWithStride",MaxPoolWithStride},
                {"lantern::derivative::AvgPoolWithStride",AvgPoolWithStride}
            };
            return map;
        }

        /**
         * @brief Get loss Map, this is to get loss function from string
         * @return std::unordered_map<std::string, std::function<af::array(const af::array&, const af::array&)>>&
         * @ingroup LanternFunction
         */
        inline std::unordered_map<std::string, std::function<af::array(const af::array&, const af::array&)>>& GetDerivativeLossMaps() {
            static std::unordered_map<std::string, std::function<af::array(const af::array&, const af::array&)>> map = {
                {"lantern::derivative::SumSquareResidual", SumSquareResidual},
                {"lantern::derivative::CrossEntropy", CrossEntropy}
            };
            return map;
        }

        /**
         * @brief Get activation Map, this is to get activation function from string
         * @return std::unordered_map<std::string, std::function<af::array(const af::array&)>>&
         * @ingroup LanternFunction
         */
        inline std::unordered_map<std::string, std::function<af::array(const af::array&)>>& GetDerivativeActivationMaps() {
            static std::unordered_map<std::string, std::function<af::array(const af::array&)>> map = {
                {"lantern::derivative::Sigmoid",Sigmoid},
                {"lantern::derivative::Swish",Swish},
                {"lantern::derivative::Linear",Linear}
            };
            return map;
        }
        

    }

}