#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "../Headers/Config.h"
#include "../Base/BaseLayer.h"
#include "FFNNode.h"

/**
 * @defgroup LanternLayer A layer definiton of lantern
 */
namespace lantern {

    namespace ffn {
        /*
            * =====================================================================
              Meta data interface

                {
                  "name": "model_name",
                  "layer_size": [],
                  "node_type_of_layer": [
                    "type"
                  ]
                }

            * =====================================================================
        */
        namespace layer {
    
            /**
             * @brief FFN Layer
             * @ingroup LanternLayer
             */
            class Layer : public BaseLayer{
            private:
                
                nlohmann::json m_meta_data = {
                    {"name", "FFN"},
                    {"layer_size", nlohmann::json::array()},
                    {"node_type_of_layer", nlohmann::json::array()}
                };
                lantern::utility::Vector<lantern::ffn::node::NodeType> m_NodeTypeOfLayer;

                lantern::utility::Vector<af::array> m_parameters;
                lantern::utility::Vector<af::array> m_prev_gradient;
                lantern::utility::Vector<af::array> m_outputs;
    
            public:
    
                Layer(){}
                Layer(Layer&& _prev_layer) noexcept {
                    this->m_LayersSize = (*_prev_layer.GetAllLayerSizes());
                    this->m_NodeTypeOfLayer = (*_prev_layer.GetAllNodeTypeOfLayer());
                }

                /**
                 * @brief Generate meta data to save model
                 */
                void GenerateMetaData() {
                    this->m_meta_data["layer_size"] = this->m_LayersSize;
                }

                /**
                 * @brief Get json Meta Data from layer
                 * @return std::string
                 */
                std::string GetMetaDataAsString() {
                    return this->m_meta_data.dump(1);
                }
    
                /**
                 * @brief Add new node into layer
                 * @tparam nodeTypeOfLayer 
                 * @param _total_node 
                 */
                template <
                    lantern::ffn::node::NodeType nodeTypeOfLayer = lantern::ffn::node::NodeType::NOTHING
                >
                void Add(uint32_t _total_node){
                    this->m_meta_data["node_type_of_layer"].push_back(
                        lantern::ffn::node::GetNodeTypeAsString(nodeTypeOfLayer)
                    );
                    this->m_LayersSize.push_back(_total_node);
                    this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
                }

                lantern::utility::Vector<af::array>* GetPrevradient() {
                    return &this->m_prev_gradient;
                }

                lantern::utility::Vector<af::array>* GetOutputs() {
                    return &this->m_outputs;
                }

                lantern::utility::Vector<af::array>* GetParameters() {
                    return &this->m_parameters;
                }
    
                /**
                 * @brief Get pointer of node type from all layer
                 * @return lantern::utility::Vector<lantern::ffn::node::NodeType>*
                 */
                lantern::utility::Vector<lantern::ffn::node::NodeType>* GetAllNodeTypeOfLayer() {
                    return &this->m_NodeTypeOfLayer;
                }
    
                /**
                 * @brief Get total node at layer
                 * @param _layer 
                 * @return uint32_t
                 */
                uint32_t GetTotalNodeAtLayer(const uint32_t& _layer) const {
                    return this->m_LayersSize[_layer];
                }

                /**
                 * @brief Print all info about layer
                 */
                void PrintLayerInfo() {
                    uint32_t convolve_index_ = 0;
                    uint32_t pooling_index_ = 0;
                    uint32_t index_ = 0;

                    for (const uint32_t& layer_size_ : m_LayersSize) {
                        lantern::utility::Vector<std::string> lines_;

                        // Add layer information
                        lines_.push_back(std::format(" Layer : {}", index_));
                        lines_.push_back(std::format(" Type : {}", lantern::ffn::node::GetNodeTypeAsString(this->m_NodeTypeOfLayer[index_])));
                        lines_.push_back(std::format(" Total Node : {}", layer_size_));

                        // Add convolution info if applicable

                        std::println("+{:-^{}}+", "", 70);
                        for (const auto& line_ : lines_){
                            std::println("|{:<{}}|", line_, 70);
                        }
                        std::println("+{:-^{}}+", "", 70);

                        index_++;
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