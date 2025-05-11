#pragma once
#include "../pch.h"
#include "Node.h"
#include "Layering.h"
#include "FeedForward.h"
#include "Function.h"
#include "Backpropagation.h"
#include "Optimizer/Optimizer.h"
#include "Initialize.h"
#include "Regularization.h"
#include "DataProcessing.h"
#include "../Headers/Vector.h"
#include "../Headers/Logging.h"

namespace lantern {

    namespace feedforward {

        class FeedForwardNetwork{
        private:

            lantern::utility::Vector<af::array> parameters;
            lantern::utility::Vector<af::array> prev_gradient;
            lantern::utility::Vector<af::array> outputs;
            lantern::utility::Vector<uint32_t> batch_index;

            af::array *input_data, *target_data;
            lantern::layer::Layer *layer;
            lantern::utility::Vector<uint32_t> each_class_size;

            uint32_t epoch = 1000;
            uint32_t current_iter = 0;
            double loss = 1, min_treshold = 0;
            af::array output, target_output;
            
        public:

            void SetInput(af::array* _input_data){
                this->input_data = _input_data;
            }

            void SetTarget(af::array* _target_data){
                this->target_data = _target_data;
            }

            void SetLayer(lantern::layer::Layer* _layer){
                this->layer = _layer;
            }

            void SetEachClassSize(std::initializer_list<uint32_t> _each_class_size){
                this->each_class_size = _each_class_size;
            }

            void SetMinimumTreshold(const double& treshold){
                this->min_treshold = treshold;
            }

            void SetEpoch(const uint32_t& epoch){
                this->epoch = epoch;
            }

            template <
                uint32_t batch_size = 10,
                typename Optimizer = lantern::optimizer::AdaptiveMomentEstimation,
                typename LossFunction = double,
                typename DerivativeLoss = af::array,
                typename OutFunction = af::array
            >
            void Train(
                Optimizer& _optimizer,
                std::function<LossFunction(af::array&, af::array&)> _loss_func,
                std::function<DerivativeLoss(af::array&, af::array&)> _derivative_loss,
                std::function<OutFunction(af::array&)> _output_func = lantern::activation::Linear
            ){

                lantern::feedforward::Initialize(
                    (*this->layer),
                    this->parameters,
                    this->prev_gradient,
                    this->outputs,
                    _optimizer
                );

                // add an empty array for gradient from loss function
                this->prev_gradient.push_back(af::array());
                uint32_t total_size_of_class = 0;
                for(auto& _size: this->each_class_size){
                    total_size_of_class += _size;
                }
	            lantern::data::GetRandomSampleClassIndex<batch_size>(this->batch_index,this->each_class_size,total_size_of_class);

                uint32_t _bacth_size = batch_size;
                while(this->current_iter < this->epoch){

                    for(auto& selected_index : batch_index){
            
                        outputs[0] = (*this->input_data).row(selected_index).T();
                        target_output = (*this->target_data).row(selected_index).T();
                        lantern::feedforward::FeedForward(
                            (*this->layer),
                            this->outputs,
                            this->parameters
                        );
            
                        output = _output_func(outputs.back());
                        loss = _loss_func(output, target_output) / batch_size;
                        std::cout << "Loss : " << loss << '\n';

                        prev_gradient.back() = _derivative_loss(output, target_output);
                        lantern::backprop::Backpropagate(
                            (*this->layer),
                            this->parameters,
                            this->prev_gradient,
                            this->outputs,
                            _optimizer,
                            _bacth_size
                        );
                    
                    }
            
                    lantern::data::GetRandomSampleClassIndex<batch_size>(this->batch_index,this->each_class_size,total_size_of_class);
                    this->current_iter++;
            
                    if(loss <= this->min_treshold){
                        break;
                    }
                }


            }

            template <typename OutFunction>
            void Predict(
                const af::array& _inputs, 
                af::array& _results,
                std::function<OutFunction(af::array&)> _out_function
            ){
                for(uint32_t i = 0; i < _inputs.dims(0); i++){
                    outputs[0] = _inputs.row(i).T();
                    lantern::feedforward::FeedForward(
                        (*this->layer),
                        this->outputs,
                        this->parameters
                    );
                    if(_results.isempty()){
                        _results = _out_function(outputs.back()).T();
                    }else{
                        _results = af::join(
                            0,
                            _results,
                            _out_function(outputs.back()).T()
                        );
                    }
                }
            }

        };

    }

}
