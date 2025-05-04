#pragma once
#include "../pch.h"

namespace lantern {
    namespace node {

        enum class NodeType{
            NOTHING, // 0
            LINEAR, // 1
            SIGMOID, // 2
            RELU, // 3
            TANH, // 4
            SWISH // 5
        };

        std::string GetNodeTypeAsString(NodeType node) {
            switch(static_cast<uint32_t>(node)){
                case 0:
                return "lantern::node::NodeType::NOTHING";
                case 1:
                return "lantern::node::NodeType::LINEAR";
                case 2:
                return "lantern::node::NodeType::SIGMOID";
                case 3:
                return "lantern::node::NodeType::RELU";
                case 4:
                return "lantern::node::NodeType::TANH";
                case 5:
                return "lantern::node::NodeType::SOFTMAX";
                case 6:
                return "lantern::node::NodeType::SWISH";
                default:
                return "UNKNOWN LANTERN NODE TYPE!";
            }
        }
        
    }
}
