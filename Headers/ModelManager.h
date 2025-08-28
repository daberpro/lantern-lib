#pragma once
#include "../pch.h"
#include "File.h"
#include "Vector.h"
#include "../ConvolutionalNeuralNetwork/CNNLayer.h"
#include "../FeedForwardNetwork/FFNLayer.h"

/**
 * @defgroup LanternModelManager A Manager for save and load model
 */

namespace lantern {

    namespace modelmanager {

        /**
         * @brief Lantern model manager, to manage model save and load
         * @ingroup LanternModelManager
         */
        class ModelManager {
        private:
            lantern::file::LanternHDF5 m_file;
            std::unordered_map<std::string, lantern::BaseLayer*> m_layers;

            /**
             * @brief Add meta data from layer pointer
             * @tparam T 
             * @param _layer 
             */
            template <typename T>
            void AddMetaData(T* _layer) {
                std::string parent_group_ = this->m_file.GetActiveGroupNameAsString();
                std::string group_name_ = parent_group_, layer_metadata_ = parent_group_ + "/LAYER_METADATA";
                H5::StrType str_type_(H5::PredType::C_S1, H5T_VARIABLE);

                this->m_file.CreateScalarDataSpace(layer_metadata_);
                this->m_file.CreateDataset(
                    "Layer_Meta_Data",
                    layer_metadata_,
                    str_type_
                );
                this->m_file.WriteDataset(
                    "Layer_Meta_Data",
                    _layer->GetMetaDataAsString(),
                    str_type_
                );


            }

        public:

            ModelManager(){}
            ModelManager(const std::string& _path) {
                this->m_file = lantern::file::LanternHDF5(_path);
            }

            /**
             * @brief Load file from HDF5 file
             * @param _path 
             */
            void LoadFile(const std::string& _path){
                this->m_file = lantern::file::LanternHDF5(_path);
            }

            /**
             * @brief Create file if file doesnot exists and replace if already exists
             */
            void Create() {
                this->m_file.Create();
            }

            /**
             * @brief Get all data such as datasets,groups,attributes from file
             */
            void GetAllData() {
                this->m_file.GetAllData();
            }

            /**
             * @brief Get out function name from file
             * @param _model_name 
             * @return 
             */
            std::string GetOutFuncName(const std::string& _model_name){
                if (this->m_file.CheckGroupExists(_model_name)) {

                    H5::StrType _str_type(H5::PredType::C_S1,H5T_VARIABLE);
                    std::string func_name_;
                    
                    this->m_file.ReadAttributeAtGroup(
                        _model_name,
                        "OUTPUT_FUNCTION",
                        _str_type,
                        func_name_
                    );

                    return func_name_;

                }else {
                    throw std::runtime_error(
                        std::format("Error ModelManager, \"{}\" uknown model name!\n", _model_name)
                    );
                }
            }

            /**
             * @brief Load params already added from file
             * @param _model_name 
             * @param _params_name 
             * @param _params 
             */
            void LoadParams(const std::string& _model_name, const std::string& _params_name, lantern::utility::Vector<af::array>& _params) {

                if (this->m_file.CheckGroupExists(_model_name)) {
                    if (this->m_file.CheckGroupExists(_model_name + "/Parameters")) {

                        uint32_t total_params_[1];
                        this->m_file.ReadAttributeAtGroup(
                            _model_name + "/Parameters",
                            _params_name + "_TOTAL",
                            H5::PredType::NATIVE_UINT32,
                            total_params_
                        );


                        for (uint32_t i = 0; i < total_params_[0]; i++) {
                            if (this->m_file.CheckDataSetExists(_model_name + "/Parameters/" + _params_name + "_" + std::to_string(i))) {
                                auto dims_ = this->m_file.GetDatasetDims(
                                    _model_name + "/Parameters/" + _params_name + "_" + std::to_string(i)
                                );
                                
                                af::dim4 af_dims_(1,1,1,1);
                                uint32_t total_elements_ = 1;
                                for (uint32_t i = 0; i < dims_.size(); i++) {
                                    af_dims_[i] = dims_[i];
                                    total_elements_ *= dims_[i];
                                }
                                
                                lantern::utility::Vector<double> temp_container_(total_elements_);
                                this->m_file.ReadDataset(
                                    _model_name + "/Parameters/" + _params_name + "_" + std::to_string(i),
                                    temp_container_.data(),
                                    H5::PredType::NATIVE_DOUBLE
                                );

                                _params.push_back(
                                    af::array(
                                        af_dims_,
                                        temp_container_.data()
                                    )
                                );
                            }
                            else {
                                throw std::runtime_error(
                                    std::format("Error ModelManager, \"{}\" uknown params!\n", _model_name)
                                );
                            }

                        }

                        
                    }
                    else {
                        throw std::runtime_error(
                            std::format("Error ModelManager, \"{}\" model was detected but no Parameters subgroup!\n", _model_name)
                        );
                    }
                }
                else {
                    throw std::runtime_error(
                        std::format("Error ModelManager, \"{}\" uknown model name!\n", _model_name)
                    );
                }

            }

            /**
             * @brief Load model and set parameters to pointer layer
             * @tparam T 
             * @param _model_name 
             * @param _layer 
             */
            template <typename T = lantern::BaseLayer>
            void LoadModel(const std::string& _model_name, T* _layer) {
                // get all groups first
                std::string raw_layer_meta_data_;
                H5::StrType _str_type(H5::PredType::C_S1,H5T_VARIABLE);
                
                if (this->m_file.CheckGroupExists(_model_name)) {
                    if (this->m_file.CheckDataSetExists(_model_name+"/Layer_Meta_Data")) {
                            
                        this->m_file.ReadDataset(_model_name + "/Layer_Meta_Data", raw_layer_meta_data_, _str_type);
                        nlohmann::json layer_meta_data_ = nlohmann::json::parse(raw_layer_meta_data_);

                        if constexpr (std::is_same_v<T, lantern::cnn::layer::Layer>) {

                            lantern::cnn::layer::Layer& CNN_layer_ = (*_layer);

                            auto& convolves_info_ = (*CNN_layer_.GetAllConvolveLayerInfo());
                            auto& layer_sizes_ = (*CNN_layer_.GetAllLayerSizes());
                            auto& poolings_info_ = (*CNN_layer_.GetAllPoolingLayerInfo());
                            auto& node_types_ = (*CNN_layer_.GetAllNodeTypeOfLayer());
                            auto& input_size_ = (*CNN_layer_.GetInputSize());

                            convolves_info_ = layer_meta_data_["convolve_layer_info"].get<lantern::utility::Vector<lantern::cnn::layer::ConvolveLayerInfo>>();
                            poolings_info_ = layer_meta_data_["pooling_layer_info"].get<lantern::utility::Vector<lantern::cnn::layer::PoolingLayerInfo>>();
                            layer_sizes_ = layer_meta_data_["layer_size"].get<lantern::utility::Vector<uint32_t>>();
                            node_types_ = layer_meta_data_["node_type_of_layer"].get<lantern::utility::Vector<lantern::cnn::node::NodeType>>();
                            input_size_ = layer_meta_data_["input_size"].get<lantern::utility::Vector<uint32_t>>();
                        }

                        if constexpr (std::is_same_v<T, lantern::ffn::layer::Layer>) {

                            lantern::ffn::layer::Layer& FFN_layer_ = (*_layer);

                            auto& all_sizes_ = (*FFN_layer_.GetAllLayerSizes());
                            auto& node_types_ = (*FFN_layer_.GetAllNodeTypeOfLayer());

                            node_types_ = layer_meta_data_["node_type_of_layer"].get<lantern::utility::Vector<lantern::ffn::node::NodeType>>();
                            all_sizes_ = layer_meta_data_["layer_size"].get<lantern::utility::Vector<uint32_t>>();
                        }
                    }
                    else {
                        throw std::runtime_error(
                            std::format("Error ModelManager, \"{}\" layer detected but has no Dataset Layer_Meta_Data!\n", _model_name)
                        );
                    }
                }else {
                    throw std::runtime_error(
                        std::format("Error ModelManager, \"{}\" uknown model name!\n", _model_name)
                    );
                }
            }

            /**
             * @brief Set out function name to file
             * @param OutFuncName 
             */
            void SetOutFunctionName(const std::string& _OutFuncName) {
                std::string parent_group_ = this->m_file.GetActiveGroupNameAsString();
                std::string group_name_ = parent_group_, attr_name_ = parent_group_ + "/OUTPUT_FUNCTION";
                H5::StrType str_type_(H5::PredType::C_S1, H5T_VARIABLE);

                this->m_file.CreateScalarDataSpace(attr_name_);
                this->m_file.CreateAttributeAtGroup(
                    group_name_,
                    attr_name_,
                    "OUTPUT_FUNCTION",
                    str_type_
                );
                this->m_file.WriteAttributeAtGroup(
                    group_name_,
                    "OUTPUT_FUNCTION",
                    str_type_,
                    _OutFuncName
                );
            }

            /*
            * @brief Add model
            * @tparam T
            */
            template <typename T>
            void AddModel(const std::string& _model_name, T* _layer) {
                this->m_file.CreateGroup(_model_name);
                this->m_layers.insert({_model_name,_layer});
            }

            /*
            * @brief Select model to get affected by all current operation after selected
            */
            void SelectModelToModify(const std::string& _model_name) {
                this->m_file.SetActiveGroup(_model_name);
            }
            
            /*
            * @brief Add params and save it into dataset with name {_params_name}_{index_of_data}
            * @param _params_name
            * @param _params
            */
            void AddParams(const std::string& _params_name,lantern::utility::Vector<af::array>& _params){
                
                std::string parent_group_ = this->m_file.GetActiveGroupNameAsString();
                lantern::BaseLayer* model_layer_ = this->m_layers.at(parent_group_);

                // create group params if not exists
                if (!this->m_file.CheckGroupExists(parent_group_ + "/Parameters")){
                    this->m_file.CreateGroup(parent_group_ + "/Parameters");
                }

                this->m_file.SetActiveGroup(parent_group_+"/Parameters");
                uint32_t rank_ = 1;
                std::string params_names_;
                std::string output_node_type_;
                double* data_ = nullptr;
                H5::StrType str_type_(H5::PredType::C_S1, H5T_VARIABLE);

                // add attribute to know total params
                this->m_file.CreateDataSpace<1>(parent_group_ + "/Parameters" + _params_name + "_TOTAL", { 1 });
                this->m_file.CreateAttributeAtGroup(
                    parent_group_ + "/Parameters",
                    parent_group_ + "/Parameters" + _params_name + "_TOTAL",
                    _params_name + "_TOTAL",
                    H5::PredType::NATIVE_UINT32
                );
                uint32_t total_param_[] = { _params.size() };
                this->m_file.WriteAttributeAtGroup(
                    parent_group_ + "/Parameters",
                    _params_name + "_TOTAL",
                    H5::PredType::NATIVE_UINT32,
                    total_param_
                );

                // check if the layer was cnn or ffn
                if (typeid(*model_layer_) == typeid(lantern::cnn::layer::Layer)) {
                    auto* layer_ = reinterpret_cast<lantern::cnn::layer::Layer*>(model_layer_);
                    auto* all_node_types_ = layer_->GetAllNodeTypeOfLayer();

                    if (_params.size() != layer_->GetAllConvolveLayerInfo()->size()) {
                        throw std::runtime_error("Error cannot add params, total node in layer lantern::cnn::layer::Layer are miss match with parameters");
                    }

                    for (uint32_t index_of_data = 0; index_of_data < _params.size(); index_of_data++) {
                        params_names_ = _params_name + "_" + std::to_string(index_of_data);
                        output_node_type_ = "OUTPUT_FUNCTION_SPACE_"+ _params_name + "_" + std::to_string(index_of_data);
                        af::array& param_ = _params[index_of_data];
                        rank_ = param_.numdims();

                        data_ = param_.host<double>();

                        this->m_file.CreateDataSpace(params_names_, rank_, {
                            static_cast<uint64_t>(param_.dims(0)),
                            static_cast<uint64_t>(param_.dims(1)),
                            static_cast<uint64_t>(param_.dims(2)),
                            static_cast<uint64_t>(param_.dims(3))
                        });
                        this->m_file.CreateDataset(
                            params_names_,
                            params_names_,
                            H5::PredType::NATIVE_DOUBLE
                        );
                        this->m_file.WriteDataset(
                            params_names_,
                            data_,
                            H5::PredType::NATIVE_DOUBLE
                        );

                       
                        af::freeHost(data_);
                    }

                    if (!this->m_file.CheckDataSetExists(parent_group_+"/Layer_Meta_Data")) {
                        this->m_file.SetActiveGroup(parent_group_);
                        this->AddMetaData(
                            layer_
                        );
                    }
                    
                }

                // check if the layer was cnn or ffn
                if (typeid(*model_layer_) == typeid(lantern::ffn::layer::Layer)) {
                    auto* layer_ = reinterpret_cast<lantern::ffn::layer::Layer*>(model_layer_);
                    auto* all_node_types_ = layer_->GetAllNodeTypeOfLayer();

                    if (_params.size() != layer_->GetAllLayerSizes()->size() - 1) {
                        throw std::runtime_error("Error cannot add params, total node in layer lantern::ffn::layer::Layer are miss match with parameters");
                    }
                    
                    for (uint32_t index_of_data = 0; index_of_data < _params.size(); index_of_data++) {
                        params_names_ = _params_name + "_" + std::to_string(index_of_data);
                        output_node_type_ = "OUTPUT_FUNCTION_SPACE_" + _params_name + "_" + std::to_string(index_of_data);
                        af::array& param_ = _params[index_of_data];
                        rank_ = param_.numdims();
                        data_ = param_.host<double>();

                        this->m_file.CreateDataSpace(params_names_, rank_, {
                            static_cast<uint64_t>(param_.dims(0)),
                            static_cast<uint64_t>(param_.dims(1)),
                            static_cast<uint64_t>(param_.dims(2)),
                            static_cast<uint64_t>(param_.dims(3))
                        });
                        this->m_file.CreateDataset(
                            params_names_,
                            params_names_,
                            H5::PredType::NATIVE_DOUBLE
                        );
                        this->m_file.WriteDataset(
                            params_names_,
                            data_,
                            H5::PredType::NATIVE_DOUBLE
                        );

                       
                        af::freeHost(data_);
                    }
                    if (!this->m_file.CheckDataSetExists(parent_group_ + "/Layer_Meta_Data")) {
                        
                        this->m_file.SetActiveGroup(parent_group_);
                        this->AddMetaData(
                            layer_
                        );
                    }
                }

                this->m_file.SetActiveGroup(parent_group_);
                
            }

        };

    }

}