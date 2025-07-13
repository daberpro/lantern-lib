#pragma once
#include "../pch.h"
#include "../Headers/Function.h"
#include "../Headers/Initialize.h"
#include "../Headers/Vector.h"
#include "../Headers/Logging.h"
#include "../Headers/File.h"

#include "Node.h"
#include "Layer.h"
#include "FeedForward.h"
#include "Backpropagation.h"
#include "Optimizer/Optimizer.h"
#include "Regularization.h"
#include "DataProcessing.h"

namespace lantern {

    namespace feedforward {

        class FeedForwardNetwork{
        private:
            lantern::utility::Vector<lantern::utility::Vector<double>> loaded_model_params;
            lantern::utility::Vector<af::array> parameters;
            lantern::utility::Vector<af::array> prev_gradient;
            lantern::utility::Vector<af::array> outputs;
            lantern::utility::Vector<uint32_t> batch_index;

            af::array *input_data, *target_data;
            lantern::ffn::layer::Layer *layer;
            lantern::utility::Vector<uint32_t> each_class_size;

            bool is_loaded_model = false;

            uint32_t epoch = 1000;
            uint32_t current_iter = 0;
            double loss = 1, min_treshold = 0;
            af::array output, target_output;

            void CheckAllRequirements() {
                if (this->input_data == nullptr) {
                    throw std::runtime_error("Input data cannot be empty");
                }

                if (this->target_data == nullptr) {
                    throw std::runtime_error("Output data cannot be empty");
                }

                if (this->layer == nullptr) {
                    throw std::runtime_error("Layer cannot be empty");
                }

                if (this->each_class_size.empty()) {
                    throw std::runtime_error("Each class size was set to 0, no input/output data will use");
                }

                if (this->min_treshold == 0) {
                    throw std::runtime_error("Treshold was set to 0, might made the model divergen");
                }

                if (this->epoch == 0) {
                    throw std::runtime_error("Epoch was set to 0, no training will run");
                }
            }
            
        public:

            FeedForwardNetwork() : input_data(nullptr), target_data(nullptr), layer(nullptr), each_class_size(NULL), min_treshold(NULL), epoch(NULL) {}
            FeedForwardNetwork(
                af::array* _input,
                af::array* _output,
                lantern::ffn::layer::Layer* _layer,
                std::initializer_list<uint32_t> _each_class_size,
                const double& _treshold,
                const uint32_t& _epoch
            ) : input_data(_input), target_data(_output), layer(_layer), each_class_size(_each_class_size), min_treshold(_treshold), epoch(_epoch){}

            lantern::utility::Vector<af::array> GetParameters(){
                return this->parameters;
            }

            lantern::utility::Vector<af::array> GetOutputsEachLayer() {
                return this->outputs;
            }

            void SetInput(af::array* _input_data){
                this->input_data = _input_data;
            }

            void SetTarget(af::array* _target_data){
                this->target_data = _target_data;
            }

            void SetLayer(lantern::ffn::layer::Layer* _layer){
                this->layer = _layer;
            }

            void SetEachClassSize(std::initializer_list<uint32_t> _each_class_size){
                this->each_class_size = _each_class_size;
            }

            void SetMinimumTreshold(const double& _treshold){
                this->min_treshold = _treshold;
            }

            void SetEpoch(const uint32_t& _epoch){
                this->epoch = _epoch;
            }

            /// <summary>
            /// Training network 
            /// </summary>
            /// <typeparam name="Optimizer"></typeparam>
            /// <typeparam name="LossFunction"></typeparam>
            /// <typeparam name="DerivativeLoss"></typeparam>
            /// <typeparam name="OutFunction"></typeparam>
            /// <typeparam name="batch_size"></typeparam>
            /// <param name="_optimizer">, Set optimizer from lantern::optimizer namespace</param>
            /// <param name="_loss_func">, Set loss function from lantern::loss:: namespace</param>
            /// <param name="_derivative_loss">, Set derivative loss from lantern::derivative namespace</param>
            /// <param name="_output_func">, Set out function from lantern::activation or lantern::probability namespace</param>
            template <
                uint32_t batch_size = 10,
                typename Optimizer = lantern::optimizer::AdaptiveMomentEstimation,
                typename LossFunction = std::function<double(af::array& output, af::array& target)>,
                typename DerivativeLoss = std::function<af::array(af::array&)>,
                typename OutFunction = std::function<af::array(af::array&, af::array&)>
            >
            void Train(
                Optimizer& _optimizer,
                LossFunction _loss_func,
                DerivativeLoss _derivative_loss,
                OutFunction _output_func = lantern::activation::Linear
            ){

                // check all required data 
                try {
                    this->CheckAllRequirements();
                }
                catch (std::runtime_error& err) {
                    std::cerr << err.what() << '\n';
                    exit(EXIT_FAILURE);
                }

                lantern::ffn::feedforward::Initialize(
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

                double progress = 0;
                int actual_progress = 0;
                uint32_t _bacth_size = batch_size;
                
                std::cout << '\n' << af::infoString() << "\n\n";
                std::cout << "Total epoch : " << this->epoch << '\n';
                
                while(this->current_iter < this->epoch){

                    for(auto& selected_index : batch_index){
            
                        outputs[0] = (*this->input_data).row(selected_index).T();
                        target_output = (*this->target_data).row(selected_index).T();
                        lantern::ffn::feedforward::FeedForward(
                            (*this->layer),
                            this->outputs,
                            this->parameters
                        );
            
                        output = _output_func(outputs.back());
                        loss = _loss_func(output, target_output) / batch_size;

                        prev_gradient.back() = _derivative_loss(output, target_output);
                        lantern::ffn::backprop::Backpropagate(
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

                    progress = static_cast<double>(this->current_iter)/static_cast<double>(this->epoch);
                    actual_progress = static_cast<int>(progress * 30);

                    std::cout << "\rTraining Progress ["
                    << std::string(actual_progress, '=')
                    << ">"
                    << std::string(30 - actual_progress, ' ')
                    << "]"
                    << std::fixed << std::setprecision(16) << std::setw(10)
                    << " Loss : " << loss
                    << " Epoch : " 
                    << this->current_iter << std::flush;
                    
                    if(loss <= this->min_treshold){
                        std::cout << "\rTraining Progress ["
                        << std::string(30, '=')
                        << ">"
                        << "]"
                        << std::fixed << std::setprecision(16) << std::setw(10)
                        << " Loss : " << loss
                        << " Epoch : " 
                        << this->current_iter << std::flush;
                        break;
                    }
                }

                std::cout << "\n\n";


            }

            /// <summary>
            /// Predict the given dataset from the training results
            /// </summary>
            /// <typeparam name="OutFunction"></typeparam>
            /// <param name="_inputs"></param>
            /// <param name="_results"></param>
            /// <param name="_out_function"></param>
            template <typename OutFunction = std::function<af::array(af::array&)>>
            void Predict(
                const af::array& _inputs, 
                af::array& _results,
                OutFunction _out_function
            ){
                for(uint32_t i = 0; i < _inputs.dims(0); i++){
                    this->outputs[0] = _inputs.row(i).T();
                    lantern::ffn::feedforward::FeedForward(
                        (*this->layer),
                        this->outputs,
                        this->parameters
                    );
                    if(_results.isempty()){
                        _results = _out_function(this->outputs.back()).T();
                    }else{
                        _results = af::join(
                            0,
                            _results,
                            _out_function(this->outputs.back()).T()
                        );
                    }
                }
            }

            /// <summary>
            /// Save Model to path with output function name, default value of output function name is "lantern::activation::Linear"
            /// </summary>
            /// <param name="_path"></param>
            /// <param name="OutFuncName"></param>
            void SaveModel(const std::string& _path,const std::string& OutFuncName = "lantern::activation::Linear") {
                
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
                lantern::utility::Vector<lantern::ffn::node::NodeType>* all_layer_type_ = this->layer->GetAllNodeTypeOfLayer();

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

                    // create dataset for param
                    model_saver_.CreateDataset(dataset_name_, dataspace_name_, H5::PredType::NATIVE_DOUBLE);
                    model_saver_.WriteDataset(dataset_name_, data, H5::PredType::NATIVE_DOUBLE);

                    // create attribute to save node type on this layer
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
                        lantern::ffn::node::GetNodeTypeAsString((*all_layer_type_)[index+1])
                    );

                    index++;
                    af::freeHost(data);

                }

                // create group model meta data to load later
                model_saver_.CreateGroup("/ModelMetaData");
                model_saver_.SetActiveGroup("/ModelMetaData");

                // get current each layer size
                auto* total_node_each_layer = this->layer->GetAllLayerSizes();

                // create attribute to save all layer size
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

                // create attribute to save output function
                // just like in Train or Predict function
                model_saver_.CreateScalarDataSpace("OUTPUT_FUNCTION");
                model_saver_.CreateAttributeAtGroup(
                    "/ModelMetaData",
                    "OUTPUT_FUNCTION",
                    "OUTPUT_FUNCTION",
                    strType
                );
                model_saver_.WriteAttributeAtGroup(
                    "/ModelMetaData",
                    "OUTPUT_FUNCTION",
                    strType,
                    OutFuncName
                );

            }

            void LoadModel(const std::string& _path){

                // celar prev params (only happend when after training and save model then load again the model)
                this->parameters.clear();
                this->outputs.clear();
                this->is_loaded_model = true;
                
                lantern::file::LanternHDF5 model_loader_(_path);
                model_loader_.GetAllData();

                // Get all metadata for model
                // Layer information
                std::string group_name_ = "/ModelMetaData";
                std::string output_model_;
                H5::StrType strType(H5::PredType::C_S1,H5T_VARIABLE);
                model_loader_.ReadAttributeAtGroup(group_name_, "OUTPUT_FUNCTION", strType, output_model_);


                // create layer object for each layer meta data
                lantern::ffn::layer::Layer* layer_ = new lantern::ffn::layer::Layer();
                auto all_node_type_ = layer_->GetAllNodeTypeOfLayer();
                auto all_layer_size_ = layer_->GetAllLayerSizes();

                // get all node size
                auto raw_dims_ = model_loader_.GetAttrDimsAtGroup(group_name_, "TOTAL_NODE_EACH_LAYER");
                uint32_t total_size_ = 1;
                for (auto dim : raw_dims_) {
                    total_size_ *= dim;
                }
                all_layer_size_->ResizeCapacity(total_size_);
                all_layer_size_->explicitTotalItem(total_size_);
                model_loader_.ReadAttributeAtGroup(group_name_, "TOTAL_NODE_EACH_LAYER", H5::PredType::NATIVE_UINT32, all_layer_size_->getData());

                // get all node type of each layer
                all_node_type_->push_back(lantern::ffn::node::NodeType::NOTHING); // first layer always an input which type of NOTHING (no activation will happend)
                std::string node_type_, layer_index_str_;
                for (uint32_t layer_index_ = 0; layer_index_ < all_layer_size_->size() - 1; layer_index_++) {

                    layer_index_str_ = std::to_string(layer_index_);
                    model_loader_.ReadAttributeAtDataset(
                        "/Parameters/L"+ layer_index_str_, 
                        "OUTPUT_FUNCTION_L" + layer_index_str_,
                        strType,
                        node_type_
                    );

                    // add an array to hold output at layer layer_index_
                    this->outputs.push_back(
                        af::constant(
                            0.0f,
                            (*all_layer_size_)[layer_index_],
                            1,
                            f64
                        )
                    );
                    // push all "layer" node type to layer
                    all_node_type_->push_back(lantern::ffn::node::GetNodeTypeFromString(node_type_));

                }
                // add the last layer outputs an array to hold
                this->outputs.push_back(
                    af::constant(
                        0.0f,
                        all_layer_size_->back(),
                        1,
                        f64
                    )
                );
                this->layer = layer_;
                layer_ = nullptr;

                // get all layer weights and bias
                layer_index_str_ = "";
                uint32_t prev_layer, current_layer;
                for (uint32_t layer_index_ = 0; layer_index_ < all_layer_size_->size() - 1; layer_index_++) {
                    
                    layer_index_str_ = std::to_string(layer_index_);
                    prev_layer = (*all_layer_size_)[layer_index_+1];
                    current_layer = (*all_layer_size_)[layer_index_];

                    this->loaded_model_params.emplace_back(prev_layer * (current_layer + 1));

                    model_loader_.ReadDataset(
                        "/Parameters/L" + layer_index_str_,
                        this->loaded_model_params.back().getData(),
                        H5::PredType::NATIVE_DOUBLE
                    );

                    af::array params(prev_layer, current_layer + 1, this->loaded_model_params.back().getData());
                    this->parameters.push_back(std::move(params));
                    
                }
            }

            ~FeedForwardNetwork(){
                // if the model layer came from loaded model
                // we need to release it, use condition to check 
                // if the layer was from loaded we can safe to delete them
                // if not do not delete them, it will cause an error
                if (this->is_loaded_model) {
                    delete this->layer;
                }
            }

            lantern::ffn::layer::Layer* GetLayer() {
                return this->layer;
            }

        };

    }

}
