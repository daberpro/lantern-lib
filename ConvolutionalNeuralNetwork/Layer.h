#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "Node.h"

namespace lantern {

    namespace cnn {
        namespace layer {

            class ConvolveLayerInfo {
            public:

                uint32_t kernel_size, padding_size, stride_size, kernel_depth;
                ConvolveLayerInfo() : kernel_size(0), padding_size(0), stride_size(0), kernel_depth(0) {}
                ConvolveLayerInfo(const uint32_t& _kernel_size,const uint32_t& _padding_size,const uint32_t& _stride_size,const uint32_t& _kernel_depth) :
                kernel_size(_kernel_size), padding_size(_padding_size) , stride_size(_stride_size), kernel_depth(_kernel_depth) {}

            };

            class PoolingLayerInfo {
            public:
                
                uint32_t size_w, size_h;
                uint32_t stride_w, stride_h;
                PoolingLayerInfo(const uint32_t& _size_w,const uint32_t& _size_h, const uint32_t& _stride_w, const uint32_t& _stride_h) : size_w(_size_w), size_h(_size_h), stride_w(_stride_w), stride_h(_stride_h){} 

            };
    
            class Layer {
            private:
    
                lantern::utility::Vector<uint32_t> m_LayersSize;
                lantern::utility::Vector<ConvolveLayerInfo> m_ConvolveLayerInfo;
                lantern::utility::Vector<PoolingLayerInfo> m_PoolingLayerInfo;
                lantern::utility::Vector<lantern::cnn::node::NodeType> m_NodeTypeOfLayer;
    
            public:
    
                Layer(){}
                Layer(Layer&& _prev_layer) {
                    this->m_LayersSize.copyPtrData(*_prev_layer.GetAllLayerSizes());
                    this->m_NodeTypeOfLayer.copyPtrData(*_prev_layer.GetAllNodeTypeOfLayer());
                    _prev_layer.~Layer();
                }
    
                void AddConvolve(
                    const uint32_t& _total_node, 
                    const uint32_t& _padding_size,
                    const uint32_t& _stride_size,
                    const uint32_t& _kernel_size,
                    const uint32_t& _kernel_depth
                ){

                    if (_total_node == 0 || _stride_size== 0 || _kernel_size == 0 || _kernel_depth == 0) {
                        throw std::runtime_error(std::format("Any parameter at function AddConvolve() cannot be zero except padding! at Layer [{}]\n", this->m_LayersSize.size()));
                    }

                    this->m_LayersSize.push_back(_total_node);
                    this->m_ConvolveLayerInfo.emplace_back(_kernel_size,_padding_size,_stride_size, _kernel_depth);
                    this->m_NodeTypeOfLayer.push_back(lantern::cnn::node::NodeType::CONVOLVE);
                }

                void AddMaxPool(
                    const uint32_t& _size_w = 2, 
                    const uint32_t& _size_h = 2,
                    const uint32_t& _stride_w = 2,
                    const uint32_t& _stride_h = 2
                ){
                    
                    this->m_LayersSize.push_back(1);
                    this->m_PoolingLayerInfo.emplace_back(_size_w,_size_h,_stride_w, _stride_h);
                    this->m_NodeTypeOfLayer.push_back(lantern::cnn::node::NodeType::MAX_POOL);

                }

                template <
                    lantern::cnn::node::NodeType nodeTypeOfLayer = lantern::cnn::node::NodeType::NOTHING
                >
                void Add(){
                    this->m_LayersSize.push_back(1);
                    this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
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
                            const auto& [kernel_size, padding_size, stride_size, kernel_depth] = this->m_ConvolveLayerInfo[convolve_index];
                            lines.push_back(std::format(" Total Weights : {}", layer_size_));
                            lines.push_back(std::format(" Total Bias : {}", layer_size_));
                            lines.push_back(" Convolve Info:");
                            lines.push_back(std::format("   - Kernel Size : {}", kernel_size));
                            lines.push_back(std::format("   - Padding     : {}", padding_size));
                            lines.push_back(std::format("   - Stride      : {}", stride_size));
                            lines.push_back(std::format("   - Depth       : {}", kernel_depth));
                            convolve_index++;
                        }

                        if (this->m_NodeTypeOfLayer[index] == lantern::cnn::node::NodeType::MAX_POOL) {
                            const auto& [width, height, stride_w, stride_h] = this->m_PoolingLayerInfo[pooling_index];
                            lines.push_back(" Max Pooling Info:");
                            lines.push_back(std::format("   - width         : {}", width));
                            lines.push_back(std::format("   - height        : {}", height));
                            lines.push_back(std::format("   - stride width  : {}", stride_w));
                            lines.push_back(std::format("   - stride height : {}", stride_h));
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