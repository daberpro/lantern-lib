#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "../Headers/Config.h"
#include "../Base/BaseLayer.h"
#include "CNNNode.h"

namespace lantern {


    namespace cnn {
        namespace layer {

            /**
             * @brief Convolution info 
             * @ingroup LanternLayer
             */
            class ConvolveLayerInfo {
            public:

                af::dim4 m_padding;
                af::dim4 m_stride;
                uint32_t m_kernel_size, m_kernel_depth;
                ConvolveLayerInfo(const uint32_t& _kernel_size,const af::dim4& _padding,const af::dim4& _stride,const uint32_t& _kernel_depth) :
                m_kernel_size(_kernel_size), m_padding(_padding), m_stride(_stride), m_kernel_depth(_kernel_depth) {}
                ConvolveLayerInfo() : m_kernel_size(0), m_kernel_depth(0) {}

            };

            /**
             * @brief Pooling info
             * @ingroup LanternLayer
             */
            class PoolingLayerInfo {
            public:
                
                af::dim4 m_stride;
                uint32_t m_size_w, m_size_h;
                PoolingLayerInfo(const uint32_t& _size_w,const uint32_t& _size_h, const af::dim4& _stride) : m_size_w(_size_w), m_size_h(_size_h), m_stride(_stride){} 

            };

            /*
            * =====================================================================
              Meta data interface
            
                {
                    "name": "model_name",
                    "layer_size": [],
                    "convolve_layer_info": [
                    {
                        "padding": [0,0,0,0],
                        "stride": [0,0,0,0],
                        "kernel_size": 0,
                        "kernel_depth": 0
                    }
                    ],
                    "pooling_layer_info": [
                    {
                        "stride": [0,0,0,0],
                        "size_w": 0,
                        "size_h": 0
                    }
                    ],
                    "node_type_of_layer": [
                    "type"
                    ],
                    "input_size": [0,0,0]
                }

            * =====================================================================
            */

            /**
             * @brief CNN Layer
             * @ingroup LanternLayer
             */
            class Layer : public BaseLayer {
            private:

                nlohmann::json m_meta_data = {
                    {"name", "CNN"},
                    {"layer_size", nlohmann::json::array()},
                    {"convolve_layer_info", nlohmann::json::array()},
                    {"pooling_layer_info", nlohmann::json::array()},
                    {"node_type_of_layer", nlohmann::json::array()},
                    {"input_size", nlohmann::json::array()}
                };
    
                lantern::utility::Vector<ConvolveLayerInfo> m_ConvolveLayerInfo;
                lantern::utility::Vector<PoolingLayerInfo> m_PoolingLayerInfo;
                lantern::utility::Vector<lantern::cnn::node::NodeType> m_NodeTypeOfLayer;
                lantern::utility::Vector<uint32_t> m_InputSize;

                lantern::utility::Vector<af::array> m_weights;
                lantern::utility::Vector<af::array> m_bias;
                lantern::utility::Vector<af::array> m_prev_gradient;
                lantern::utility::Vector<af::array> m_outputs;

                // temp container to save pooling stride modification result
                // because the function activation pooling with stride just change actual input
                // into new shape and we want to save that modify input
                
                lantern::utility::Vector<af::array> m_PoolingModificationInputResult;
                lantern::utility::Vector<af::array> m_BatchNormDerivativeForParams;
                lantern::utility::Vector<af::array> m_BatchNormDerivativeForOutputs;

                lantern::utility::Vector<af::array> m_BatchNormParams;
    
            public:
    
                Layer(){}
                Layer(Layer&& _prev_layer) noexcept {
                    this->m_LayersSize = (*_prev_layer.GetAllLayerSizes());
                    this->m_NodeTypeOfLayer = (*_prev_layer.GetAllNodeTypeOfLayer());
                }

                void operator =(Layer&& _prev_layer) noexcept {
                    this->m_LayersSize = (*_prev_layer.GetAllLayerSizes());
                    this->m_NodeTypeOfLayer = (*_prev_layer.GetAllNodeTypeOfLayer());
                }
    
                /**
                 * @brief Add new concolve node into layer
                 * @param _total_node 
                 * @param _padding 
                 * @param _stride 
                 * @param _kernel_size 
                 * @param _kernel_depth 
                 */
                void AddConvolve(
                    const uint32_t& _total_node = 1, 
                    const af::dim4& _padding = af::dim4((dim_t)0,(dim_t)0),
                    const af::dim4& _stride = af::dim4(1,1),
                    const uint32_t& _kernel_size = 1,
                    const uint32_t& _kernel_depth = 1
                ){

                    if (_total_node == 0 || _stride.elements() == 0 || _kernel_size == 0 || _kernel_depth == 0) {
                        throw std::runtime_error(std::format("Any parameter at function AddConvolve() cannot be zero except padding! at Layer [{}]\n", this->m_LayersSize.size()));
                    }

                    // add convolve info to meta data
                    this->m_meta_data["convolve_layer_info"].push_back({
                        {"padding",_padding},
                        {"stride",_stride},
                        {"kernel_size", _kernel_size},
                        {"kernel_depth", _kernel_depth},
                    });

                    this->m_meta_data["node_type_of_layer"].push_back(
                        lantern::cnn::node::GetNodeTypeAsString(lantern::cnn::node::NodeType::CONVOLVE)
                    );

                    this->m_LayersSize.push_back(_total_node);
                    this->m_ConvolveLayerInfo.emplace_back(_kernel_size,_padding,_stride, _kernel_depth);
                    this->m_NodeTypeOfLayer.push_back(lantern::cnn::node::NodeType::CONVOLVE);
                }

                /**
                 * @brief Add pooling node into layer
                 * @tparam POOL_TYPE 
                 * @param _size_w 
                 * @param _size_h 
                 * @param _stride 
                 */
                template <lantern::cnn::node::NodeType POOL_TYPE = lantern::cnn::node::NodeType::AVG_POOL>
                void AddPool(
                    const uint32_t& _size_w = 2, 
                    const uint32_t& _size_h = 2,
                    const af::dim4& _stride = af::dim4(1,1)
                ){
                    
                    this->m_meta_data["pooling_layer_info"].push_back({
                        {"stride", _stride},
                        {"size_w", _size_w},
                        {"size_h", _size_h}
                    });
                    this->m_meta_data["node_type_of_layer"].push_back(
                        lantern::cnn::node::GetNodeTypeAsString(POOL_TYPE)
                    );
                    this->m_LayersSize.push_back(1);
                    this->m_PoolingLayerInfo.emplace_back(_size_w,_size_h,_stride);
                    this->m_NodeTypeOfLayer.push_back(POOL_TYPE);

                }

                /**
                 * @brief Add ndoe to layer
                 * @tparam nodeTypeOfLayer 
                 */
                template <
                    lantern::cnn::node::NodeType nodeTypeOfLayer = lantern::cnn::node::NodeType::NOTHING
                >
                void Add(){
                    this->m_meta_data["node_type_of_layer"].push_back(
                        lantern::cnn::node::GetNodeTypeAsString(nodeTypeOfLayer)
                    );
                    this->m_LayersSize.push_back(1);
                    this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
                }

                /**
                 * @brief Set input size with sizes {w,h,d,n}
                 * @param _input_sizes 
                 */
                void SetInputSize(lantern::utility::Vector<uint32_t>&& _input_sizes){
                    this->m_meta_data["input_size"] = _input_sizes;
                    this->m_InputSize = std::move(_input_sizes);
                }

                lantern::utility::Vector<af::array>* GetBatchNormDerivativeParams(){
                    return &this->m_BatchNormDerivativeForParams;
                }

                lantern::utility::Vector<af::array>* GetBatchNormParams(){
                    return &this->m_BatchNormParams;
                }

                lantern::utility::Vector<af::array>* GetBatchNormDerivativeOutputs() {
                    return &this->m_BatchNormDerivativeForOutputs;
                }

                lantern::utility::Vector<af::array>* GetWeights(){
                    return &this->m_weights;
                }

                lantern::utility::Vector<af::array>* GetBias(){
                    return &this->m_bias;
                }

                lantern::utility::Vector<af::array>* GetPrevGradient(){
                    return &this->m_prev_gradient;
                }
                
                lantern::utility::Vector<af::array>* GetOutputs(){
                    return &this->m_outputs;
                }

                /**
                 * @brief Get pointer of input sizes
                 * @return lantern::utility::Vector<uint32_t>*
                 */
                lantern::utility::Vector<uint32_t>* GetInputSize(){
                    return &this->m_InputSize;
                }

                /**
                 * @brief Get pointer of all convolve node
                 * @return lantern::utility::Vector<ConvolveLayerInfo>*
                 */
                lantern::utility::Vector<ConvolveLayerInfo>* GetAllConvolveLayerInfo() {
                    return &this->m_ConvolveLayerInfo;
                }

                /**
                 * @brief Get pointer of all pooling node
                 * @return lantern::utility::Vector<PoolingLayerInfo>*
                 */
                lantern::utility::Vector<PoolingLayerInfo>* GetAllPoolingLayerInfo() {
                    return &this->m_PoolingLayerInfo;
                }
    
                /**
                 * @brief Get all type of node from layer
                 * @return lantern::utility::Vector<lantern::cnn::node::NodeType>*
                 */
                lantern::utility::Vector<lantern::cnn::node::NodeType>* GetAllNodeTypeOfLayer() {
                    return &this->m_NodeTypeOfLayer;
                }

                /**
                 * @brief Get pooling node output for backpropagation
                 * @return lantern::utility::Vector<af::array>*
                 */
                lantern::utility::Vector<af::array>* GetPoolingModificationInputResult() {
                    return &this->m_PoolingModificationInputResult;
                }

                /**
                 * @brief Generate meta data to save model
                 */
                void GenerateMetaData() {
                    this->m_meta_data["layer_size"] = this->m_LayersSize;
                }

                /**
                 * @brief Get meta data as string json
                 * @return std::string
                 */
                std::string GetMetaDataAsString() {
                    return this->m_meta_data.dump(1);
                }

                /**
                 * @brief Get pointer to meta data
                 * @return nlohmann::json*
                 */
                nlohmann::json* GetMetaDataPtr() {
                    return &this->m_meta_data;
                }

                /**
                 * @brief Print layer info
                 */
                void PrintLayerInfo() {
                    uint32_t convolve_index_ = 0;
                    uint32_t pooling_index_ = 0;
                    uint32_t index_ = 0;

                    for (const uint32_t& layer_size_ : m_LayersSize) {
                        lantern::utility::Vector<std::string> lines;

                        // Add layer information
                        lines.push_back(std::format(" Layer : {}", index_));
                        lines.push_back(std::format(" Type : {}", lantern::cnn::node::GetNodeTypeAsString(this->m_NodeTypeOfLayer[index_])));

                        // Add convolution info if applicable
                        switch (this->m_NodeTypeOfLayer[index_])
                        {
                        case lantern::cnn::node::NodeType::CONVOLVE: {
                            const auto& [padding_, stride_, kernel_size_, kernel_depth_] = this->m_ConvolveLayerInfo[convolve_index_];
                            lines.push_back(std::format(" Total Weights : {}", layer_size_));
                            lines.push_back(std::format(" Total Bias : {}", layer_size_));
                            lines.push_back(" Convolve Info:");
                            lines.push_back(std::format("   - Kernel Size    : {}", kernel_size_));
                            lines.push_back(std::format("   - Padding        : {}", padding_));
                            lines.push_back(std::format("   - Stride Width   : {}", stride_));
                            lines.push_back(std::format("   - Depth          : {}", kernel_depth_));
                            convolve_index_++;
                            break;
                        }
                        case lantern::cnn::node::NodeType::MAX_POOL:
                        case lantern::cnn::node::NodeType::AVG_POOL: {
                            const auto& [stride_, width_, height_] = this->m_PoolingLayerInfo[pooling_index_];
                            lines.push_back(" Max Pooling Info:");
                            lines.push_back(std::format("   - width         : {}", width_));
                            lines.push_back(std::format("   - height        : {}", height_));
                            lines.push_back(std::format("   - stride width  : {}", stride_));
                            pooling_index_++;
                            break;
                        }
                        }


                        std::println("+{:-^{}}+", "", 70);
                        for (const auto& line : lines){
                            std::println("|{:<{}}|", line, 70);
                        }
                        std::println("+{:-^{}}+", "", 70);

                        index_++;
                    }
                }

    
                /**
                 * @brief Get total node at layer
                 * @param _layer 
                 * @return uint32_t
                 */
                uint32_t GetTotalNodeAtLayer(const uint32_t& _layer) const {
                    return this->m_LayersSize[_layer];
                }
    
                ~Layer() = default;
    
            };

    
        }
    }


}

namespace nlohmann
{
    template <>
    struct adl_serializer<lantern::cnn::layer::PoolingLayerInfo>
    {
        static lantern::cnn::layer::PoolingLayerInfo from_json(const json& _j)
        {
            lantern::cnn::layer::PoolingLayerInfo pooling_info_(
                _j["size_w"].get<uint32_t>(),
                _j["size_h"].get<uint32_t>(),
                _j["stride"].get<af::dim4>()
            );
            return pooling_info_;
        }

        static void to_json(json& _j,const lantern::cnn::layer::PoolingLayerInfo& _pooling_info)
        {
            _j = {
                {"stride", _pooling_info.m_stride},
                {"size_w", _pooling_info.m_size_w},
                {"size_h", _pooling_info.m_size_h}
            };
        }
    };
    template <>
    struct adl_serializer<lantern::cnn::layer::ConvolveLayerInfo>
    {
        static lantern::cnn::layer::ConvolveLayerInfo from_json(const json& _j)
        {
            lantern::cnn::layer::ConvolveLayerInfo convolve_info_(
                _j["kernel_size"].get<uint32_t>(),
                _j["padding"].get<af::dim4>(),
                _j["stride"].get<af::dim4>(),
                _j["kernel_depth"].get<uint32_t>()
            );
            return convolve_info_;
        }

        static void to_json(json& _j,const lantern::cnn::layer::ConvolveLayerInfo& _convolve_info)
        {
            _j = {
                {"kernel_size", _convolve_info.m_kernel_size},
                {"stride", _convolve_info.m_stride},
                {"padding", _convolve_info.m_padding},
                {"kernel_depth", _convolve_info.m_kernel_depth}
            };
        }
    };

} // namespace nlohmann

template <>
struct std::formatter<lantern::utility::Vector<lantern::cnn::node::NodeType>> {

   constexpr auto parse(std::format_parse_context& _ctx) {
       return _ctx.begin();
   }

   auto format(const lantern::utility::Vector<lantern::cnn::node::NodeType>& _obj, std::format_context& _ctx) const {

       std::ostringstream oss_;
       oss_ << "[";
       for (size_t i = 0; i < _obj.size(); ++i) {
           if (i > 0) oss_ << ", ";
           oss_ << "\n " << lantern::cnn::node::GetNodeTypeAsString(_obj[i]);
       }
       oss_ << "\n]\n";
       return std::format_to(_ctx.out(), "{}", oss_.str());
   }
};