#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "FFNNode.h"

namespace lantern {

    namespace ffn {

        namespace layer {
    
            class Layer {
            private:
    
                lantern::utility::Vector<uint32_t> m_LayersSize;
                lantern::utility::Vector<lantern::ffn::node::NodeType> m_NodeTypeOfLayer;
    
            public:
    
                Layer(){}
                Layer(Layer&& _prev_layer) {
                    this->m_LayersSize.copyPtrData(*_prev_layer.GetAllLayerSizes());
                    this->m_NodeTypeOfLayer.copyPtrData(*_prev_layer.GetAllNodeTypeOfLayer());
                    _prev_layer.~Layer();
                }
    
                template <
                    lantern::ffn::node::NodeType nodeTypeOfLayer = lantern::ffn::node::NodeType::NOTHING
                >
                void Add(uint32_t _total_node){
                    this->m_LayersSize.push_back(_total_node);
                    this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
                }
    
                lantern::utility::Vector<uint32_t>* GetAllLayerSizes() {
                    return &this->m_LayersSize;
                }
    
                lantern::utility::Vector<lantern::ffn::node::NodeType>* GetAllNodeTypeOfLayer() {
                    return &this->m_NodeTypeOfLayer;
                }
    
                uint32_t GetTotalNodeAtLayer(const uint32_t& _layer) const {
                    return this->m_LayersSize[_layer];
                }

                void PrintLayerInfo() {
                    uint32_t convolve_index = 0;
                    uint32_t pooling_index = 0;
                    uint32_t index = 0;

                    for (const uint32_t& layer_size_ : m_LayersSize) {
                        lantern::utility::Vector<std::string> lines;

                        // Add layer information
                        lines.push_back(std::format(" Layer : {}", index));
                        lines.push_back(std::format(" Type : {}", lantern::ffn::node::GetNodeTypeAsString(this->m_NodeTypeOfLayer[index])));
                        lines.push_back(std::format(" Total Node : {}", layer_size_));

                        // Add convolution info if applicable

                        std::println("+{:-^{}}+", "", 70);
                        for (const auto& line : lines){
                            std::println("|{:<{}}|", line, 70);
                        }
                        std::println("+{:-^{}}+", "", 70);

                        index++;
                    }
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
struct std::formatter<lantern::utility::Vector<lantern::ffn::node::NodeType>> {

   constexpr auto parse(std::format_parse_context& ctx) {
       return ctx.begin();
   }

   auto format(const lantern::utility::Vector<lantern::ffn::node::NodeType>& obj, std::format_context& ctx) const {

       std::ostringstream oss;
       oss << "[";
       for (size_t i = 0; i < obj.size(); ++i) {
           if (i > 0) oss << ", ";
           oss << '\n' << lantern::ffn::node::GetNodeTypeAsString(obj[i]);
       }
       oss << "\n]\n";
       return std::format_to(ctx.out(), "{}", oss.str());
   }
};