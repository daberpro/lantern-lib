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

#define LANTERN_GET_FUNC_NAME(Func) #Func

namespace lantern{
    
    namespace activation {

        af::array Sigmoid(const af::array& value){
            return 1/(1 + af::exp(-value));
        }

        af::array Linear(const af::array& value){
            return value;
        }

        af::array ReLU(af::array& value){
            value(af::where(value <= 0)) = 0;
            return value;
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
        af::array MaxPool(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w) {

            int32_t dims_0_, dims_1_, dims_2_;
            af::array res_ = af::reorder(_input, 0, 2, 1);
            res_ = af::moddims(res_, res_.dims(0), res_.dims(1) * res_.dims(2), 1);
            dims_0_ = res_.dims(0);
            dims_1_ = res_.dims(1);
            dims_2_ = res_.dims(2);
            res_ = af::moddims(res_, _pool_h, res_.dims(1) / _pool_h, res_.dims(0));
            res_ = af::reorder(res_,0,2,1);
            res_ = af::moddims(res_,_pool_h,_pool_w,(res_.dims(1) * res_.dims(0)) / (_pool_w * _pool_h) * res_.dims(2));
            // _sliced_derivative = res_;
            res_ = af::max(res_,1);
            res_ = af::max(res_,0);
            res_ = af::reorder(res_, 2, 1, 0);

            // // Get derivative for this pooling
            // _sliced_derivative = (_sliced_derivative == af::tile(af::reorder(res_, 2, 1, 0), _sliced_derivative.dims(0), _sliced_derivative.dims(1))).as(f64);
            // _sliced_derivative = af::moddims(_sliced_derivative, _sliced_derivative.dims(0), _sliced_derivative.dims(1) * _sliced_derivative.dims(2));
            // _sliced_derivative = af::moddims(_sliced_derivative, _sliced_derivative.dims(0), _input.dims(1), _input.dims(1) / _pool_h * _input.dims(2));
            // _sliced_derivative = af::reorder(_sliced_derivative, 0, 2, 1);
            // _sliced_derivative = af::moddims(_sliced_derivative, _input.dims(0) , _input.dims(1), _input.dims(2));
            
            res_ = af::moddims(res_, dims_0_ / _pool_h, dims_1_ / _pool_w, 1);
            res_ = af::moddims(res_, _input.dims(0) / _pool_h, _input.dims(1) / _pool_w, _input.dims(2));

            return res_.T();
        }

        af::array MaxPoolWithStride(const af::array& _input, uint32_t const& _pool_h, uint32_t const& _pool_w, const uint32_t& _stride_w, const uint32_t& _stride_h){

            
            af::array res_ = _input;
            res_ = res_(
                af::seq(0,res_.dims(0) - 1,_stride_h),
                af::seq(0,res_.dims(1) - 1,_stride_w)
            );
            
            // Get pooling width and height
            uint32_t width_ = res_.dims(0);
            uint32_t height_ = res_.dims(1);
            uint32_t depth_ = res_.dims(2);
            uint32_t rest_width_ = width_ % _pool_w; // Get valid width of pooling 
            uint32_t rest_height_ = height_ % _pool_h; // Get valid height of 

            uint32_t padding_width_ = (rest_width_ == 0) ? 0 : (_pool_w - rest_width_); // Get the required size to create valid pooling from rest width
            uint32_t padding_height_ = (rest_height_ == 0) ? 0 : (_pool_h - rest_height_);  // Get the required size to create valid pooling from rest height

            res_ = af::pad(
                res_,
                af::dim4(0,0,0,0),
                af::dim4(padding_width_,padding_height_,0,0),
                AF_PAD_ZERO
            );

            af::print("Padding", res_);

            return MaxPool(
                res_,
                _pool_h,
                _pool_w
            );

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
        
        af::array MaxPool(const af::array& _input,const af::array& _outputs, const uint32_t& _pool_h, const uint32_t& _pool_w){

            af::array res_ = af::reorder(_input, 0, 2, 1);
            res_ = af::moddims(res_, res_.dims(0), res_.dims(1) * res_.dims(2), 1);
            res_ = af::moddims(res_, _pool_h, res_.dims(1) / _pool_h, res_.dims(0));
            res_ = af::reorder(res_,0,2,1);
            res_ = af::moddims(res_,_pool_h,_pool_w,(res_.dims(1) * res_.dims(0)) / (_pool_w * _pool_h) * res_.dims(2));
            res_ = (res_ == af::tile(af::reorder(_outputs, 2, 1, 0), res_.dims(0), res_.dims(1))).as(f64);
            res_ = af::moddims(res_, res_.dims(0), res_.dims(1) * res_.dims(2));
            res_ = af::moddims(res_, res_.dims(0), _input.dims(1), _input.dims(1) / _pool_h * _input.dims(2));
            res_ = af::reorder(res_, 0, 2, 1);
            res_ = af::moddims(res_, _input.dims(0) , _input.dims(1), _input.dims(2));

            return res_;
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