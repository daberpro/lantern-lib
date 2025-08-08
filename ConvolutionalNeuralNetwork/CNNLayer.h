#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "CNNNode.h"

namespace lantern {

    namespace cnn {
        namespace layer {

            class ConvolveLayerInfo {
            public:

                af::dim4 padding;
                af::dim4 stride;
                uint32_t kernel_size, kernel_depth;
                ConvolveLayerInfo(const uint32_t& _kernel_size,const af::dim4& _padding,const af::dim4& _stride,const uint32_t& _kernel_depth) :
                kernel_size(_kernel_size), padding(_padding), stride(_stride), kernel_depth(_kernel_depth) {}
                ConvolveLayerInfo() : kernel_size(0), kernel_depth(0) {}

            };

            class PoolingLayerInfo {
            public:
                
                af::dim4 stride;
                uint32_t size_w, size_h;
                PoolingLayerInfo(const uint32_t& _size_w,const uint32_t& _size_h, const af::dim4& _stride) : size_w(_size_w), size_h(_size_h), stride(_stride){} 

            };
    
            class Layer {
            private:
    
                lantern::utility::Vector<uint32_t> m_LayersSize;
                lantern::utility::Vector<ConvolveLayerInfo> m_ConvolveLayerInfo;
                lantern::utility::Vector<PoolingLayerInfo> m_PoolingLayerInfo;
                lantern::utility::Vector<lantern::cnn::node::NodeType> m_NodeTypeOfLayer;
                lantern::utility::Vector<uint32_t> m_InputSize;

                // temp container to save pooling stride modification result
                // because the function activation pooling with stride just change actual input
                // into new shape and we want to save that modify input
                
                lantern::utility::Vector<af::array> m_PoolingModificationInputResult;
    
            public:
    
                Layer(){}
                Layer(Layer&& _prev_layer){
                    this->m_LayersSize.copyPtrData(*_prev_layer.GetAllLayerSizes());
                    this->m_NodeTypeOfLayer.copyPtrData(*_prev_layer.GetAllNodeTypeOfLayer());
                    _prev_layer.~Layer();
                }
    
                void AddConvolve(
                    const uint32_t& _total_node, 
                    const af::dim4& _padding,
                    const af::dim4& _stride,
                    const uint32_t& _kernel_size,
                    const uint32_t& _kernel_depth
                ){

                    if (_total_node == 0 || _stride.elements() == 0 || _kernel_size == 0 || _kernel_depth == 0) {
                        throw std::runtime_error(std::format("Any parameter at function AddConvolve() cannot be zero except padding! at Layer [{}]\n", this->m_LayersSize.size()));
                    }

                    this->m_LayersSize.push_back(_total_node);
                    this->m_ConvolveLayerInfo.emplace_back(_kernel_size,_padding,_stride, _kernel_depth);
                    this->m_NodeTypeOfLayer.push_back(lantern::cnn::node::NodeType::CONVOLVE);
                }

                template <lantern::cnn::node::NodeType POOL_TYPE = lantern::cnn::node::NodeType::AVG_POOL>
                void AddPool(
                    const uint32_t& _size_w = 2, 
                    const uint32_t& _size_h = 2,
                    const af::dim4& _stride = af::dim4(1,1)
                ){
                    
                    this->m_LayersSize.push_back(1);
                    this->m_PoolingLayerInfo.emplace_back(_size_w,_size_h,_stride);
                    this->m_NodeTypeOfLayer.push_back(POOL_TYPE);

                }

                template <
                    lantern::cnn::node::NodeType nodeTypeOfLayer = lantern::cnn::node::NodeType::NOTHING
                >
                void Add(){
                    this->m_LayersSize.push_back(1);
                    this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
                }

                void SetInputSize(lantern::utility::Vector<uint32_t>&& _input_sizes){
                    this->m_InputSize = std::move(_input_sizes);
                }

                lantern::utility::Vector<uint32_t>* GetInputSize(){
                    return &this->m_InputSize;
                }
    
                lantern::utility::Vector<uint32_t>* GetAllLayerSizes() {
                    return &this->m_LayersSize;
                }

                lantern::utility::Vector<ConvolveLayerInfo>* GetAllConvolveLayerInfo() {
                    return &this->m_ConvolveLayerInfo;
                }

                lantern::utility::Vector<PoolingLayerInfo>* GetAllPoolingLayerInfo() {
                    return &this->m_PoolingLayerInfo;
                }
    
                lantern::utility::Vector<lantern::cnn::node::NodeType>* GetAllNodeTypeOfLayer() {
                    return &this->m_NodeTypeOfLayer;
                }

                lantern::utility::Vector<af::array>* GetPoolingModificationInputResult() {
                    return &this->m_PoolingModificationInputResult;
                }

                void PrintLayerInfo() {
                    uint32_t convolve_index = 0;
                    uint32_t pooling_index = 0;
                    uint32_t index = 0;

                    for (const uint32_t& layer_size_ : m_LayersSize) {
                        lantern::utility::Vector<std::string> lines;

                        // Add layer information
                        lines.push_back(std::format(" Layer : {}", index));
                        lines.push_back(std::format(" Type : {}", lantern::cnn::node::GetNodeTypeAsString(this->m_NodeTypeOfLayer[index])));

                        // Add convolution info if applicable
                        if (this->m_NodeTypeOfLayer[index] == lantern::cnn::node::NodeType::CONVOLVE) {
                            const auto& [padding, stride, kernel_size, kernel_depth] = this->m_ConvolveLayerInfo[convolve_index];
                            lines.push_back(std::format(" Total Weights : {}", layer_size_));
                            lines.push_back(std::format(" Total Bias : {}", layer_size_));
                            lines.push_back(" Convolve Info:");
                            lines.push_back(std::format("   - Kernel Size    : {}", kernel_size));
                            lines.push_back(std::format("   - Padding        : {}", padding));
                            lines.push_back(std::format("   - Stride Width   : {}", stride));
                            lines.push_back(std::format("   - Depth          : {}", kernel_depth));
                            convolve_index++;
                        }

                        if (this->m_NodeTypeOfLayer[index] == lantern::cnn::node::NodeType::MAX_POOL) {
                            const auto& [stride, width, height] = this->m_PoolingLayerInfo[pooling_index];
                            lines.push_back(" Max Pooling Info:");
                            lines.push_back(std::format("   - width         : {}", width));
                            lines.push_back(std::format("   - height        : {}", height));
                            lines.push_back(std::format("   - stride width  : {}", stride));
                            pooling_index++;
                        }


                        std::println("+{:-^{}}+", "", 70);
                        for (const auto& line : lines){
                            std::println("|{:<{}}|", line, 70);
                        }
                        std::println("+{:-^{}}+", "", 70);

                        index++;
                    }
                }

    
                uint32_t GetTotalNodeAtLayer(const uint32_t& _layer) const {
                    return this->m_LayersSize[_layer];
                }
    
                ~Layer() {
                    this->m_NodeTypeOfLayer.clear();
                    this->m_LayersSize.clear();
                }
    
            };
    
        }
    }


}

template <>
struct std::formatter<lantern::utility::Vector<lantern::cnn::node::NodeType>> {

   constexpr auto parse(std::format_parse_context& ctx) {
       return ctx.begin();
   }

   auto format(const lantern::utility::Vector<lantern::cnn::node::NodeType>& obj, std::format_context& ctx) const {

       std::ostringstream oss;
       oss << "[";
       for (size_t i = 0; i < obj.size(); ++i) {
           if (i > 0) oss << ", ";
           oss << "\n " << lantern::cnn::node::GetNodeTypeAsString(obj[i]);
       }
       oss << "\n]\n";
       return std::format_to(ctx.out(), "{}", oss.str());
   }
};