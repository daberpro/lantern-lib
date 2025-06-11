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
#include "../Headers/File.h"

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

            lantern::utility::Vector<af::array> GetParameters(){
                return this->parameters;
            }

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

            void SaveModel(const std::string& _path){
                
                lantern::file::LanternHDF5 model_saver_(_path);
                model_saver_.Create();
                model_saver_.GetAllData();
                
                /**
                 * Parameter info
                 * L{n} is a parameter at n layer
                 * 
                 * path "/Parameters" is a folder which contain all result of training parameters
                 * 
                */

                // first create the group or folder for hold all parameters
                model_saver_.CreateGroup("/Parameters");
                model_saver_.SetActiveGroup("/Parameters"); // then set the group to be active group

                std::string label_ = "param_";
                std::string dataspace_name_;
                std::string dataset_name_;
                std::string output_node_type_;
                uint32_t index = 0, rank = 1;
                double* data = nullptr;
                H5::StrType strType(H5::PredType::C_S1,H5T_VARIABLE);
                lantern::utility::Vector<lantern::node::NodeType>* all_layer_type_ = this->layer->GetAllNodeTypeOfLayer();

                for(af::array& param : this->parameters){

                    data = param.host<double>();
                    dataspace_name_ = label_+std::to_string(index);
                    output_node_type_ = "OUTPUT_FUNCTION_SPACE_"+std::to_string(index);
                    dataset_name_ = std::string("L") + std::to_string(index);

                    // create dataspace for each array data
                    rank = param.numdims();
                    model_saver_.CreateDataSpace(dataspace_name_,rank,{
                        static_cast<uint64_t>(param.dims(0)),
                        static_cast<uint64_t>(param.dims(1)),
                        static_cast<uint64_t>(param.dims(2)),
                        static_cast<uint64_t>(param.dims(3))
                    });

                    model_saver_.CreateDataset(dataset_name_, dataspace_name_, H5::PredType::NATIVE_DOUBLE);
                    model_saver_.PrintAllDatasets();
                    model_saver_.WriteDataset(dataset_name_, data, H5::PredType::NATIVE_DOUBLE);

                    model_saver_.CreateScalarDataSpace(output_node_type_);
                    model_saver_.CreateAttributeAtDataset(
                        dataset_name_,
                        output_node_type_,
                        "OUTPUT_FUNCTION_"+dataset_name_,
                        strType
                    );
                    model_saver_.WriteAttributeAtDataset(
                        dataset_name_,
                        "OUTPUT_FUNCTION_"+dataset_name_,
                        strType,
                        lantern::node::GetNodeTypeAsString((*all_layer_type_)[index+1])
                    );

                    index++;
                    af::freeHost(data);

                }

                model_saver_.CreateGroup("/ModelMetaData");
                model_saver_.SetActiveGroup("/ModelMetaData");

                auto* total_node_each_layer = this->layer->GetAllLayerSizes();

                model_saver_.CreateDataSpace<2>("TOTAL_NODE_EACH_LAYER",{1,total_node_each_layer->size()});
                model_saver_.CreateAttributeAtGroup(
                    "/ModelMetaData",
                    "TOTAL_NODE_EACH_LAYER",
                    "TOTAL_NODE_EACH_LAYER",
                    H5::PredType::NATIVE_UINT32
                );
                model_saver_.WriteAttributeAtGroup(
                    "/ModelMetaData",
                    "TOTAL_NODE_EACH_LAYER",
                    H5::PredType::NATIVE_UINT32,
                    total_node_each_layer->getData()
                );

                model_saver_.CreateScalarDataSpace("OUTPUT_PROBABILITY_FUNCTION");
                model_saver_.CreateAttributeAtGroup(
                    "/ModelMetaData",
                    "OUTPUT_PROBABILITY_FUNCTION",
                    "OUTPUT_PROBABILITY_FUNCTION",
                    strType
                );
                model_saver_.WriteAttributeAtGroup(
                    "/ModelMetaData",
                    "OUTPUT_PROBABILITY_FUNCTION",
                    strType,
                    lantern::node::GetNodeTypeAsString(all_layer_type_->back()) // TODO this must get probability output function 
                );


            }

            void LoadModel(const std::string& _path){
                
                
            }

        };

    }

}
