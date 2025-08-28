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
            lantern::utility::Vector<lantern::utility::Vector<double>> m_loaded_model_params;
            lantern::utility::Vector<af::array> m_parameters;
            lantern::utility::Vector<af::array> m_prev_gradient;
            lantern::utility::Vector<af::array> m_outputs;
            lantern::utility::Vector<uint32_t> m_batch_index;

            lantern::modelmanager::ModelManager m_model_manager;

            af::array *m_input_data, *m_target_data;
            lantern::ffn::layer::Layer *m_layer;
            lantern::utility::Vector<uint32_t> m_each_class_size;

            bool m_is_loaded_model = false;

            uint32_t m_epoch = 100;
            uint32_t m_current_iter = 0;
            double m_loss = 1, m_min_treshold = 0;
            af::array m_output, m_target_output;

            /**
             * @brief Chek all requirements before init network
             */
            void CheckAllRequirements() {
                if (this->m_input_data == nullptr) {
                    throw std::runtime_error("Input data cannot be empty");
                }

                if (this->m_target_data == nullptr) {
                    throw std::runtime_error("Output data cannot be empty");
                }

                if (this->m_layer == nullptr) {
                    throw std::runtime_error("Layer cannot be empty");
                }

                if (this->m_each_class_size.empty()) {
                    throw std::runtime_error("Each class size was set to 0, no input/output data will use");
                }

                if (this->m_min_treshold == 0) {
                    throw std::runtime_error("Treshold was set to 0, might made the model divergen");
                }

                if (this->m_epoch == 0) {
                    throw std::runtime_error("Epoch was set to 0, no training will run");
                }
            }
            
        public:

            FeedForwardNetwork() : m_input_data(nullptr), m_target_data(nullptr), m_layer(nullptr), m_each_class_size(NULL), m_min_treshold(NULL), m_epoch(NULL) {
                this->m_parameters = *this->m_layer->GetParameters();
                this->m_outputs = *this->m_layer->GetOutputs();
                this->m_prev_gradient = *this->m_layer->GetPrevradient();

            }
            FeedForwardNetwork(
                af::array* _input,
                af::array* _output,
                lantern::ffn::layer::Layer* _layer,
                std::initializer_list<uint32_t> _each_class_size,
                const double& _treshold,
                const uint32_t& _epoch
            ) : m_input_data(_input), m_target_data(_output), m_layer(_layer), m_each_class_size(_each_class_size), m_min_treshold(_treshold), m_epoch(_epoch){
                
            }

            /**
             * @brief Get parameters from parameters (Weights,Bias)
             * @return lantern::utility::Vector<af::array>
             */
            lantern::utility::Vector<af::array> GetParameters(){
                return this->m_parameters;
            }

            /**
             * @brief Get output of each layer
             * @return lantern::utility::Vector<af::array>
             */
            lantern::utility::Vector<af::array> GetOutputsEachLayer() {
                return this->m_outputs;
            }

            /**
             * @brief Set input to feed to network
             * @param _input_data 
             */
            void SetInput(af::array* _input_data){
                this->m_input_data = _input_data;
            }

            /**
             * @brief Set target to adjust the output fromnetwork
             * @param _target_data 
             */
            void SetTarget(af::array* _target_data){
                this->m_target_data = _target_data;
            }

            /**
             * @brief Set layer pointer
             * @param _layer 
             */
            void SetLayer(lantern::ffn::layer::Layer* _layer){
                this->m_layer = _layer;
            }

            /**
             * @brief Set each class size, this is a size of each class in dataset like 5 cats and 3 dogs become {5,3-1}, we substract the last ith 1 because the index start at 0
             * @param _each_class_size 
             */
            void SetEachClassSize(std::initializer_list<uint32_t> _each_class_size){
                this->m_each_class_size = _each_class_size;
            }

            /**
             * @brief Set minimum treshold for loss during training
             * @param _treshold 
             */
            void SetMinimumTreshold(const double& _treshold){
                this->m_min_treshold = _treshold;
            }

            /**
             * @brief Set epoch to train
             * @param _epoch 
             */
            void SetEpoch(const uint32_t& _epoch){
                this->m_epoch = _epoch;
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
                    (*this->m_layer),
                    _optimizer
                );

                // add an empty array for gradient from loss function
                this->m_prev_gradient.push_back(af::array());
                uint32_t total_size_of_class = 0;
                for(auto& _size: this->m_each_class_size){
                    total_size_of_class += _size;
                }
	            lantern::data::GetRandomSampleClassIndex<batch_size>(this->m_batch_index,this->m_each_class_size,total_size_of_class);

                double progress_ = 0;
                int actual_progress_ = 0;
                uint32_t bacth_size_ = batch_size;
                
                std::cout << '\n' << af::infoString() << "\n\n";
                std::cout << "Total epoch : " << this->m_epoch << '\n';
                
                while(this->m_current_iter < this->m_epoch){

                    for(auto& selected_index : this->m_batch_index){
            
                        this->m_outputs[0] = (*this->m_input_data).row(selected_index).T();
                        this->m_target_output = (*this->m_target_data).row(selected_index).T();
                        lantern::ffn::feedforward::FeedForward(
                            (*this->m_layer)
                        );
            
                        this->m_output = _output_func(this->m_outputs.back());
                        this->m_loss = _loss_func(this->m_output, this->m_target_output) / batch_size;

                        this->m_prev_gradient.back() = _derivative_loss(this->m_output, this->m_target_output);
                        lantern::ffn::backprop::Backpropagate(
                            (*this->m_layer),
                            _optimizer,
                            bacth_size_
                        );
                    
                    }
            
                    lantern::data::GetRandomSampleClassIndex<batch_size>(this->m_batch_index,this->m_each_class_size,total_size_of_class);
                    this->m_current_iter++;

                    progress_ = static_cast<double>(this->m_current_iter)/static_cast<double>(this->m_epoch);
                    actual_progress_ = static_cast<int>(progress_ * 30);

                    std::cout << "\rTraining Progress ["
                    << std::string(actual_progress_, '=')
                    << ">"
                    << std::string(30 - actual_progress_, ' ')
                    << "]"
                    << std::fixed << std::setprecision(16) << std::setw(10)
                    << " Loss : " << this->m_loss
                    << " Epoch : " 
                    << this->m_current_iter << std::flush;
                    
                    if(this->m_loss <= this->m_min_treshold){
                        std::cout << "\rTraining Progress ["
                        << std::string(30, '=')
                        << ">"
                        << "]"
                        << std::fixed << std::setprecision(16) << std::setw(10)
                        << " Loss : " << this->m_loss
                        << " Epoch : " 
                        << this->m_current_iter << std::flush;
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
                    this->m_outputs[0] = _inputs.row(i).T();
                    lantern::ffn::feedforward::FeedForward(
                        (*this->m_layer)
                    );
                    if(_results.isempty()){
                        _results = _out_function(this->m_outputs.back()).T();
                    }else{
                        _results = af::join(
                            0,
                            _results,
                            _out_function(this->m_outputs.back()).T()
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
                
                this->m_is_loaded_model = false;
                this->m_model_manager.LoadFile(_path);
                this->m_model_manager.Create();
                this->m_model_manager.GetAllData();

                this->m_layer->GenerateMetaData();
                this->m_model_manager.AddModel("/FFN",this->m_layer);
                this->m_model_manager.SelectModelToModify("/FFN");
                this->m_model_manager.AddParams("Weights_Bias", this->m_parameters);
                this->m_model_manager.SetOutFunctionName(OutFuncName);

            }

            /**
             * @brief Load model from HDF5
             * @param _path 
             */
            void LoadModel(const std::string& _path){

                // celar prev params (only happend when after training and save model then load again the model)
                this->m_is_loaded_model = true;
                this->m_parameters.clear();
                this->m_outputs.clear();

                this->m_model_manager.LoadFile(_path);
                this->m_model_manager.GetAllData();
                
                this->m_layer = new lantern::ffn::layer::Layer();
                this->m_model_manager.LoadModel("/FFN",this->m_layer);
                this->m_model_manager.LoadParams("/FFN","Weights_Bias",this->m_parameters);

                for(uint32_t i = 0; i < this->m_layer->GetAllLayerSizes()->size(); i++){
                    this->m_outputs.push_back(af::array());
                }
            }

            ~FeedForwardNetwork(){
                // if the model layer came from loaded model
                // we need to release it, use condition to check 
                // if the layer was from loaded we can safe to delete them
                // if not do not delete them, it will cause an error
                if (this->m_is_loaded_model) {
                    delete this->m_layer;
                }
            }

            /**
             * @brief Get layer
             * @return lantern::ffn::layer::Layer*
             */
            lantern::ffn::layer::Layer* GetLayer() {
                return this->m_layer;
            }

        };

    }

}
