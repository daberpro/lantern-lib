/*
* ================================================================================
* LANTERN-LIB
* by
* Daberdev (Ari Susanto)
* ================================================================================
* 
*/
#pragma once
#include "../pch.h"
#include "../ConvolutionalNeuralNetwork/CNNNode.h"

#define LANTERN_GET_FUNC_NAME(Func) #Func

namespace lantern{

    namespace utility {

        template <typename T>
        T max(const T& a, const T& b){
            return (a < b? b : a);
        }

    }
    
    namespace activation {

        af::array Sigmoid(const af::array& value){
            return 1/(1 + af::exp(-value));
        }

        af::array Linear(const af::array& value){
            return value;
        }

        af::array ReLU(const af::array& value){
            return af::max(0,value);
        }

        af::array LeakyReLU(const af::array& value){
            return af::max(value * 0.1, value);
        }

        af::array TanH(const af::array& value){
            af::array exp_ = af::exp(value);
            af::array nexp_ = af::exp(-value);
            return (exp_ - nexp_)/(exp_ + nexp_);
        }

        af::array Swish(const af::array& value){
            return value * Sigmoid(value);
        }

    }

    namespace pooling {
        
        /*
        =========================================================================================
        Pooling Algorithm
        Made By Daberdev (Ari Susanto)
        =========================================================================================
        */

        /**
         * =========================================================================================
         * Performs max pooling on a 3D/2D ArrayFire input using specified pooling dimensions.
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
         * =========================================================================================
         */
         af::array MaxPool(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w) {

            int32_t dims_0_, dims_1_, dims_2_;
            af::array res_ = af::reorder(_input, 0, 2, 1),_sliced_derivative;
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

        af::array AvgPool(const af::array& input, const uint32_t& pool_h,const uint32_t& pool_w) {

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

        template <lantern::cnn::node::NodeType PoolType>
        std::pair<af::array,af::array> PoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const af::dim4& _stride){
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
                    throw std::runtime_error("PoolWithStride<PoolingType>() error unknown pooling type!");
                }
            }

            return {
                result,
                valid_input_ // this is use for backpropagation of Pooling
            };
        }
        
        /**
         * =========================================================================================
         * Applies max pooling with custom stride to the input ArrayFire array.
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
         * =========================================================================================
         */
        std::pair<af::array,af::array> MaxPoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const af::dim4& _stride){
            return PoolWithStride<lantern::cnn::node::NodeType::MAX_POOL>(_input,_pool_h,_pool_w,_stride);
        }

        std::pair<af::array,af::array> AvgPoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const af::dim4& _stride) {
            return PoolWithStride<lantern::cnn::node::NodeType::AVG_POOL>(_input,_pool_h,_pool_w,_stride);
        }

    }


    namespace probability {

        af::array SoftMax(const af::array& value){
            af::array exp_ = af::exp(value);
            af::array sum_exp_ = af::sum(af::exp(value));
            return exp_/sum_exp_;
        }

    }

    namespace loss {

        double SumSquareResidual(const af::array& output, const af::array& target){
            return af::pow(target - output,2).scalar<double>();
        }

        double CrossEntropy(const af::array& output, const af::array& target){
            return af::sum(-(target * af::log(output + 1e-012))).scalar<double>();
        }

    }

    namespace derivative {

        
        af::array MaxPool(const af::array& _input, const uint32_t& _pool_h, const uint32_t& _pool_w){
           
            af::array res_ = af::reorder(_input, 0, 2, 1),_sliced_derivative;
            res_ = af::moddims(res_, res_.dims(0), res_.dims(1) * res_.dims(2), 1);
            res_ = af::moddims(res_, _pool_h, res_.dims(1) / _pool_h, res_.dims(0));
            res_ = af::reorder(res_,0,2,1);
            res_ = af::moddims(res_,_pool_h,_pool_w,(res_.dims(1) * res_.dims(0)) / (_pool_w * _pool_h) * res_.dims(2));
            _sliced_derivative = res_;
            res_ = af::max(res_,1);
            res_ = af::max(res_,0);
            res_ = af::reorder(res_, 2, 1, 0);
            _sliced_derivative = (_sliced_derivative == af::tile(af::reorder(res_, 2, 1, 0), _sliced_derivative.dims(0), _sliced_derivative.dims(1))).as(f64);
            _sliced_derivative = af::moddims(_sliced_derivative,_sliced_derivative.dims(0),_input.dims(1),(_input.dims(0) / _pool_h) * _input.dims(2));
            _sliced_derivative = af::moddims(_sliced_derivative.T(),_input.dims(0),_input.dims(1),_input.dims(2));
           
            return _sliced_derivative;

        }

        af::array MaxPoolWithStride(const af::array& _input, const af::array& _modify_input, const uint32_t& _pool_h, const uint32_t& _pool_w, const af::dim4& _stride){
            
            /**
             * Create a temporary output variabel with the size same as the input
             * in feedforward
             */
            af::array temp_out = af::constant(0.0, _input.dims(),f64);
            
            // do derivative of max pool
            af::array res_ = MaxPool(
                _modify_input,
                _pool_h,
                _pool_w
            );

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
                af::span
            ) = res_;

            return temp_out.T();
        }

        af::array Sigmoid(const af::array& value){
            return value * (1 - value);
        }

        af::array Swish(const af::array& value){
            return activation::Sigmoid(value) +  Sigmoid(value) * value;
        }

        af::array Linear(const af::array& value){
            return af::constant(1.0f,value.dims(0),value.dims(1),f64);
        }

        af::array SumSquareResidual(const af::array& output, const af::array& target){
            return -2 * (target - output);
        }

        af::array CrossEntropySoftMax(const af::array& output, const af::array& target){
            return output - target;
        }

    }

}