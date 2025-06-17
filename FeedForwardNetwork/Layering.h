#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "Node.h"

namespace lantern {

    namespace layer {

        class Layer {
        private:

            lantern::utility::Vector<uint32_t> m_LayersSize;
            lantern::utility::Vector<lantern::node::NodeType> m_NodeTypeOfLayer;

        public:

            Layer(){}
            Layer(Layer&& _prev_layer) {
                this->m_LayersSize.copyPtrData(*_prev_layer.GetAllLayerSizes());
                this->m_NodeTypeOfLayer.copyPtrData(*_prev_layer.GetAllNodeTypeOfLayer());
                _prev_layer.~Layer();
            }

            template <
                lantern::node::NodeType nodeTypeOfLayer = lantern::node::NodeType::NOTHING
            >
            void Add(uint32_t _total_node){
                this->m_LayersSize.push_back(_total_node);
                this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
            }

            void Add(uint32_t _total_node, const lantern::node::NodeType& _node_type) {
                this->m_LayersSize.push_back(_total_node);
                this->m_NodeTypeOfLayer.push_back(_node_type);
            }

            lantern::utility::Vector<uint32_t>* GetAllLayerSizes() {
                return &this->m_LayersSize;
            }

            lantern::utility::Vector<lantern::node::NodeType>* GetAllNodeTypeOfLayer() {
                return &this->m_NodeTypeOfLayer;
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