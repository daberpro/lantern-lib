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

            template <
                lantern::node::NodeType nodeTypeOfLayer = lantern::node::NodeType::NOTHING
            >
            void Add(uint32_t total_node){
                this->m_LayersSize.push_back(total_node);
                this->m_NodeTypeOfLayer.push_back(nodeTypeOfLayer);
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

        };

    }

}