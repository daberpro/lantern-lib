#pragma once

#include <string>
#include <cstring>

namespace lantern {
    namespace cnn {
        namespace node {

            enum class NodeType {
                NOTHING,      // 0
                MAX_POOL,     // 1
                CONVOLVE,    // 2
                RELU,         // 3
                AVG_POOL,     // 4
                GLOBAL_POOL,  // 5
                L2_POOL,      // 6
                FLATTEN,      // 7
                UNKNOWN       // 8
            };

            inline std::string GetNodeTypeAsString(NodeType node) {
                switch (node) {
                    case NodeType::NOTHING:      return "lantern::cnn::node::NodeType::NOTHING";
                    case NodeType::MAX_POOL:     return "lantern::cnn::node::NodeType::MAX_POOL";
                    case NodeType::CONVOLVE:    return "lantern::cnn::node::NodeType::CONVOLVE";
                    case NodeType::RELU:         return "lantern::cnn::node::NodeType::RELU";
                    case NodeType::AVG_POOL:     return "lantern::cnn::node::NodeType::AVG_POOL";
                    case NodeType::GLOBAL_POOL:  return "lantern::cnn::node::NodeType::GLOBAL_POOL";
                    case NodeType::L2_POOL:      return "lantern::cnn::node::NodeType::L2_POOL";
                    case NodeType::FLATTEN:      return "lantern::cnn::node::NodeType::FLATTEN";
                    default:                     return "UNKNOWN LANTERN NODE TYPE!";
                }
            }

            inline NodeType GetNodeTypeFromString(const std::string& _str) {
                if (_str == "lantern::cnn::node::NodeType::NOTHING")             return NodeType::NOTHING;
                else if (_str == "lantern::cnn::node::NodeType::MAX_POOL")       return NodeType::MAX_POOL;
                else if (_str == "lantern::cnn::node::NodeType::CONVOLVE")      return NodeType::CONVOLVE;
                else if (_str == "lantern::cnn::node::NodeType::RELU")           return NodeType::RELU;
                else if (_str == "lantern::cnn::node::NodeType::AVG_POOL")       return NodeType::AVG_POOL;
                else if (_str == "lantern::cnn::node::NodeType::GLOBAL_POOL")    return NodeType::GLOBAL_POOL;
                else if (_str == "lantern::cnn::node::NodeType::L2_POOL")        return NodeType::L2_POOL;
                else if (_str == "lantern::cnn::node::NodeType::FLATTEN")        return NodeType::FLATTEN;
                else                                                        return NodeType::UNKNOWN;
            }

        }
    }
}
