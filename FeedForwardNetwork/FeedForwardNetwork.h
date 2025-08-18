#pragma once
#include "../pch.h"
#include "../Headers/Function.h"
#include "../Headers/Initialize.h"
#include "../Headers/Vector.h"
#include "../Headers/DataProcessing.h"
//#include "../Headers/Logging.h"
#include "../Headers/File.h"
#include "../Headers/ModelManager.h"

#include "FFNNode.h"
#include "FFNLayer.h"
#include "FFNFeedForward.h"
#include "FFNBackpropagation.h"
#include "FFNOptimizer/FFNOptimizer.h"
#include "FFNRegularization.h"


/**
 * @defgroup LanternFFNWrapper A wrapper for FFN model
 */

namespace lantern {

    namespace feedforward {

        /**
         * @brief Feed forward network class, a wrapper for lantern ffn
         * @ingroup LanternFFNWrapper
         */
        class FeedForwardNetwork{
        private:
            lantern::utility::Vector<lantern::utility::Vector<double>> loaded_model_params;
            lantern::utility::Vector<af::array> parameters;
            lantern::utility::Vector<af::array> prev_gradient;
            lantern::utility::Vector<af::array> outputs;
            lantern::utility::Vector<uint32_t> batch_index;

            lantern::modelmanager::ModelManager model_manager;

            af::array *input_data, *target_data;
            lantern::ffn::layer::Layer *layer;
            lantern::utility::Vector<uint32_t> each_class_size;

            bool is_loaded_model = false;

            uint32_t epoch = 100;
            uint32_t current_iter = 0;
            double loss = 1, min_treshold = 0;
            af::array output, target_output;

            /**
             * @brief Chek all requirements before init network
             */
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

            /**
             * @brief Get parameters from parameters (Weights,Bias)
             * @return lantern::utility::Vector<af::array>
             */
            lantern::utility::Vector<af::array> GetParameters(){
                return this->parameters;
            }

            /**
             * @brief Get output of each layer
             * @return lantern::utility::Vector<af::array>
             */
            lantern::utility::Vector<af::array> GetOutputsEachLayer() {
                return this->outputs;
            }

            /**
             * @brief Set input to feed to network
             * @param _input_data 
             */
            void SetInput(af::array* _input_data){
                this->input_data = _input_data;
            }

            /**
             * @brief Set target to adjust the output fromnetwork
             * @param _target_data 
             */
            void SetTarget(af::array* _target_data){
                this->target_data = _target_data;
            }

            /**
             * @brief Set layer pointer
             * @param _layer 
             */
            void SetLayer(lantern::ffn::layer::Layer* _layer){
                this->layer = _layer;
            }

            /**
             * @brief Set each class size, this is a size of each class in dataset like 5 cats and 3 dogs become {5,3-1}, we substract the last ith 1 because the index start at 0
             * @param _each_class_size 
             */
            void SetEachClassSize(std::initializer_list<uint32_t> _each_class_size){
                this->each_class_size = _each_class_size;
            }

            /**
             * @brief Set minimum treshold for loss during training
             * @param _treshold 
             */
            void SetMinimumTreshold(const double& _treshold){
                this->min_treshold = _treshold;
            }

            /**
             * @brief Set epoch to train
             * @param _epoch 
             */
            void SetEpoch(const uint32_t& _epoch){
                this->epoch = _epoch;
            }

            /**
             * @brief Train the network
             * @tparam Optimizer 
             * @tparam LossFunction 
             * @tparam DerivativeLoss 
             * @tparam OutFunction 
             * @tparam batch_size 
             * @param _optimizer 
             * @param _loss_func 
             * @param _derivative_loss 
             * @param _output_func 
             */
            template <
                uint32_t batch_size = 10,
                typename Optimizer = lantern::ffn::optimizer::AdaptiveMomentEstimation,
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
                    std::cerr << "lantern::ffn::FeedForwardNetwork::Train<>(), Error " << err.what() << '\n';
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

            /**
             * @brief Predict the given input
             * @tparam OutFunction 
             * @param _inputs 
             * @param _results 
             * @param _out_function 
             */
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

            /**
             * @brief Save model with HDF5
             * @param _path 
             * @param OutFuncName 
             */
            void SaveModel(const std::string& _path,const std::string& OutFuncName = "lantern::activation::Linear") {
                
                this->is_loaded_model = false;
                this->model_manager.LoadFile(_path);
                this->model_manager.Create();
                this->model_manager.GetAllData();

                this->layer->GenerateMetaData();
                this->model_manager.AddModel("/FFN",this->layer);
                this->model_manager.SelectModelToModify("/FFN");
                this->model_manager.AddParams("Weights_Bias", this->parameters);
                this->model_manager.SetOutFunctionName(OutFuncName);

            }

            /**
             * @brief Load model from HDF5
             * @param _path 
             */
            void LoadModel(const std::string& _path){

                // celar prev params (only happend when after training and save model then load again the model)
                this->is_loaded_model = true;
                this->parameters.clean();
                this->outputs.clean();

                this->model_manager.LoadFile(_path);
                this->model_manager.GetAllData();
                
                this->layer = new lantern::ffn::layer::Layer();
                this->model_manager.LoadModel("/FFN",this->layer);
                this->model_manager.LoadParams("/FFN","Weights_Bias",this->parameters);

                for(uint32_t i = 0; i < this->layer->GetAllLayerSizes()->size(); i++){
                    this->outputs.push_back(af::array());
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

            /**
             * @brief Get layer
             * @return lantern::ffn::layer::Layer*
             */
            lantern::ffn::layer::Layer* GetLayer() {
                return this->layer;
            }

        };

    }

}
