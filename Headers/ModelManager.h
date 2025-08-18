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
            lantern::file::LanternHDF5 file;
            std::unordered_map<std::string, lantern::BaseLayer*> layers;

            /**
             * @brief Add meta data from layer pointer
             * @tparam T 
             * @param _layer 
             */
            template <typename T>
            void AddMetaData(T* _layer) {
                std::string parent_group = this->file.GetActiveGroupNameAsString();
                std::string group_name = parent_group, layer_metadata = parent_group + "/LAYER_METADATA";
                H5::StrType str_type_(H5::PredType::C_S1, H5T_VARIABLE);

                this->file.CreateScalarDataSpace(layer_metadata);
                this->file.CreateDataset(
                    "Layer_Meta_Data",
                    layer_metadata,
                    str_type_
                );
                this->file.WriteDataset(
                    "Layer_Meta_Data",
                    _layer->GetMetaDataAsString(),
                    str_type_
                );


            }

        public:

            ModelManager(){}
            ModelManager(const std::string& _path) {
                this->file = lantern::file::LanternHDF5(_path);
            }

            /**
             * @brief Load file from HDF5 file
             * @param _path 
             */
            void LoadFile(const std::string& _path){
                this->file = lantern::file::LanternHDF5(_path);
            }

            /**
             * @brief Create file if file doesnot exists and replace if already exists
             */
            void Create() {
                this->file.Create();
            }

            /**
             * @brief Get all data such as datasets,groups,attributes from file
             */
            void GetAllData() {
                this->file.GetAllData();
            }

            /**
             * @brief Get out function name from file
             * @param _model_name 
             * @return 
             */
            std::string GetOutFuncName(const std::string& _model_name){
                if (this->file.CheckGroupExists(_model_name)) {

                    H5::StrType _str_type(H5::PredType::C_S1,H5T_VARIABLE);
                    std::string func_name;
                    
                    this->file.ReadAttributeAtGroup(
                        _model_name,
                        "OUTPUT_FUNCTION",
                        _str_type,
                        func_name
                    );

                    return func_name;

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

                if (this->file.CheckGroupExists(_model_name)) {
                    if (this->file.CheckGroupExists(_model_name + "/Parameters")) {

                        uint32_t total_params[1];
                        this->file.ReadAttributeAtGroup(
                            _model_name + "/Parameters",
                            _params_name + "_TOTAL",
                            H5::PredType::NATIVE_UINT32,
                            total_params
                        );


                        for (uint32_t i = 0; i < total_params[0]; i++) {
                            if (this->file.CheckDataSetExists(_model_name + "/Parameters/" + _params_name + "_" + std::to_string(i))) {
                                auto dims = this->file.GetDatasetDims(
                                    _model_name + "/Parameters/" + _params_name + "_" + std::to_string(i)
                                );
                                
                                af::dim4 af_dims(1,1,1,1);
                                uint32_t total_elements = 1;
                                for (uint32_t i = 0; i < dims.size(); i++) {
                                    af_dims[i] = dims[i];
                                    total_elements *= dims[i];
                                }
                                
                                lantern::utility::Vector<double> temp_container(total_elements);
                                this->file.ReadDataset(
                                    _model_name + "/Parameters/" + _params_name + "_" + std::to_string(i),
                                    temp_container.getData(),
                                    H5::PredType::NATIVE_DOUBLE
                                );

                                _params.push_back(
                                    af::array(
                                        af_dims,
                                        temp_container.getData()
                                    )
                                );
                            }
                            else {
                                for (auto& [key, value] : (*this->file.GetDatasetsPtr())) {
                                    std::println("{}", key);
                                }
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
                std::string raw_layer_meta_data;
                H5::StrType _str_type(H5::PredType::C_S1,H5T_VARIABLE);
                
                if (this->file.CheckGroupExists(_model_name)) {
                    if (this->file.CheckDataSetExists(_model_name+"/Layer_Meta_Data")) {
                            
                        this->file.ReadDataset(_model_name + "/Layer_Meta_Data", raw_layer_meta_data, _str_type);
                        nlohmann::json layer_meta_data = nlohmann::json::parse(raw_layer_meta_data);

                        if constexpr (std::is_same_v<T, lantern::cnn::layer::Layer>) {

                            lantern::cnn::layer::Layer& CNN_layer = (*_layer);

                            auto& convolves_info = (*CNN_layer.GetAllConvolveLayerInfo());
                            auto& layer_sizes = (*CNN_layer.GetAllLayerSizes());
                            auto& poolings_info = (*CNN_layer.GetAllPoolingLayerInfo());
                            auto& node_types = (*CNN_layer.GetAllNodeTypeOfLayer());
                            auto& input_size = (*CNN_layer.GetInputSize());

                            convolves_info = layer_meta_data["convolve_layer_info"].get<lantern::utility::Vector<lantern::cnn::layer::ConvolveLayerInfo>>();
                            poolings_info = layer_meta_data["pooling_layer_info"].get<lantern::utility::Vector<lantern::cnn::layer::PoolingLayerInfo>>();
                            layer_sizes = layer_meta_data["layer_size"].get<lantern::utility::Vector<uint32_t>>();
                            node_types = layer_meta_data["node_type_of_layer"].get<lantern::utility::Vector<lantern::cnn::node::NodeType>>();
                            input_size = layer_meta_data["input_size"].get<lantern::utility::Vector<uint32_t>>();
                        }

                        if constexpr (std::is_same_v<T, lantern::ffn::layer::Layer>) {

                            lantern::ffn::layer::Layer& FFN_layer = (*_layer);

                            auto& all_sizes = (*FFN_layer.GetAllLayerSizes());
                            auto& node_types = (*FFN_layer.GetAllNodeTypeOfLayer());

                            node_types = layer_meta_data["node_type_of_layer"].get<lantern::utility::Vector<lantern::ffn::node::NodeType>>();
                            all_sizes = layer_meta_data["layer_size"].get<lantern::utility::Vector<uint32_t>>();
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
            void SetOutFunctionName(const std::string& OutFuncName) {
                std::string parent_group = this->file.GetActiveGroupNameAsString();
                std::string group_name = parent_group, attr_name = parent_group + "/OUTPUT_FUNCTION";
                H5::StrType str_type_(H5::PredType::C_S1, H5T_VARIABLE);

                this->file.CreateScalarDataSpace(attr_name);
                this->file.CreateAttributeAtGroup(
                    group_name,
                    attr_name,
                    "OUTPUT_FUNCTION",
                    str_type_
                );
                this->file.WriteAttributeAtGroup(
                    group_name,
                    "OUTPUT_FUNCTION",
                    str_type_,
                    OutFuncName
                );
            }

            /*
            * @brief Add model
            * @tparam T
            */
            template <typename T>
            void AddModel(const std::string& _model_name, T* _layer) {
                this->file.CreateGroup(_model_name);
                this->layers.insert({_model_name,_layer});
            }

            /*
            * @brief Select model to get affected by all current operation after selected
            */
            void SelectModelToModify(const std::string& _model_name) {
                this->file.SetActiveGroup(_model_name);
            }
            
            /*
            * @brief Add params and save it into dataset with name {_params_name}_{index_of_data}
            * @param _params_name
            * @param _params
            */
            void AddParams(const std::string& _params_name,lantern::utility::Vector<af::array>& _params){
                
                std::string parent_group = this->file.GetActiveGroupNameAsString();
                lantern::BaseLayer* model_layer = this->layers.at(parent_group);

                // create group params if not exists
                if (!this->file.CheckGroupExists(parent_group + "/Parameters")){
                    this->file.CreateGroup(parent_group + "/Parameters");
                }

                this->file.SetActiveGroup(parent_group+"/Parameters");
                uint32_t rank = 1;
                std::string params_names;
                std::string output_node_type_;
                double* data = nullptr;
                H5::StrType str_type_(H5::PredType::C_S1, H5T_VARIABLE);

                // add attribute to know total params
                this->file.CreateDataSpace<1>(parent_group + "/Parameters" + _params_name + "_TOTAL", { 1 });
                this->file.CreateAttributeAtGroup(
                    parent_group + "/Parameters",
                    parent_group + "/Parameters" + _params_name + "_TOTAL",
                    _params_name + "_TOTAL",
                    H5::PredType::NATIVE_UINT32
                );
                uint32_t total_param[] = { _params.size() };
                this->file.WriteAttributeAtGroup(
                    parent_group + "/Parameters",
                    _params_name + "_TOTAL",
                    H5::PredType::NATIVE_UINT32,
                    total_param
                );

                // check if the layer was cnn or ffn
                if (typeid(*model_layer) == typeid(lantern::cnn::layer::Layer)) {
                    auto* layer = reinterpret_cast<lantern::cnn::layer::Layer*>(model_layer);
                    auto* all_node_types = layer->GetAllNodeTypeOfLayer();

                    if (_params.size() != layer->GetAllConvolveLayerInfo()->size()) {
                        throw std::runtime_error("Error cannot add params, total node in layer lantern::cnn::layer::Layer are miss match with parameters");
                    }

                    for (uint32_t index_of_data = 0; index_of_data < _params.size(); index_of_data++) {
                        params_names = _params_name + "_" + std::to_string(index_of_data);
                        output_node_type_ = "OUTPUT_FUNCTION_SPACE_"+ _params_name + "_" + std::to_string(index_of_data);
                        af::array& param = _params[index_of_data];
                        rank = param.numdims();

                        data = param.host<double>();

                        this->file.CreateDataSpace(params_names, rank, {
                            static_cast<uint64_t>(param.dims(0)),
                            static_cast<uint64_t>(param.dims(1)),
                            static_cast<uint64_t>(param.dims(2)),
                            static_cast<uint64_t>(param.dims(3))
                        });
                        this->file.CreateDataset(
                            params_names,
                            params_names,
                            H5::PredType::NATIVE_DOUBLE
                        );
                        this->file.WriteDataset(
                            params_names,
                            data,
                            H5::PredType::NATIVE_DOUBLE
                        );

                       
                        af::freeHost(data);
                    }

                    if (!this->file.CheckDataSetExists(parent_group+"/Layer_Meta_Data")) {
                        this->file.SetActiveGroup(parent_group);
                        this->AddMetaData(
                            layer
                        );
                    }
                    
                }

                // check if the layer was cnn or ffn
                if (typeid(*model_layer) == typeid(lantern::ffn::layer::Layer)) {
                    auto* layer = reinterpret_cast<lantern::ffn::layer::Layer*>(model_layer);
                    auto* all_node_types = layer->GetAllNodeTypeOfLayer();

                    if (_params.size() != layer->GetAllLayerSizes()->size() - 1) {
                        throw std::runtime_error("Error cannot add params, total node in layer lantern::ffn::layer::Layer are miss match with parameters");
                    }
                    
                    for (uint32_t index_of_data = 0; index_of_data < _params.size(); index_of_data++) {
                        params_names = _params_name + "_" + std::to_string(index_of_data);
                        output_node_type_ = "OUTPUT_FUNCTION_SPACE_" + _params_name + "_" + std::to_string(index_of_data);
                        af::array& param = _params[index_of_data];
                        rank = param.numdims();

                        data = param.host<double>();

                        this->file.CreateDataSpace(params_names, rank, {
                            static_cast<uint64_t>(param.dims(0)),
                            static_cast<uint64_t>(param.dims(1)),
                            static_cast<uint64_t>(param.dims(2)),
                            static_cast<uint64_t>(param.dims(3))
                        });
                        this->file.CreateDataset(
                            params_names,
                            params_names,
                            H5::PredType::NATIVE_DOUBLE
                        );
                        this->file.WriteDataset(
                            params_names,
                            data,
                            H5::PredType::NATIVE_DOUBLE
                        );

                       
                        af::freeHost(data);
                    }
                    if (!this->file.CheckDataSetExists(parent_group + "/Layer_Meta_Data")) {
                        
                        this->file.SetActiveGroup(parent_group);
                        this->AddMetaData(
                            layer
                        );
                    }
                }

                this->file.SetActiveGroup(parent_group);
                
            }

        };

    }

}