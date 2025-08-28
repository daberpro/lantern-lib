#pragma once

#include <string>
#include <cstring>

namespace lantern {
    namespace cnn {
        namespace node {

            /**
             * @brief Node type of CNN
             * @ingroup LanternNode
             */
            enum class NodeType {
                NOTHING,      // 0
                MAX_POOL,     // 1
                CONVOLVE,    // 2
                RELU,         // 3
                AVG_POOL,     // 4
                GLOBAL_AVG_POOL,  // 5
                L2_POOL,      // 6
                FLATTEN,      // 7
                UNKNOWN,       // 8
                LEAKY_RELU,
                SWISH,
                SIGMOID,
                BATCH_NORM
            };

            /**
             * @brief Get node type as string
             * @param node 
             * @return std::string
             */
            inline std::string GetNodeTypeAsString(NodeType _node) {
                switch (_node) {
                    case NodeType::NOTHING:      return "lantern::cnn::node::NodeType::NOTHING";
                    case NodeType::MAX_POOL:     return "lantern::cnn::node::NodeType::MAX_POOL";
                    case NodeType::CONVOLVE:    return "lantern::cnn::node::NodeType::CONVOLVE";
                    case NodeType::RELU:         return "lantern::cnn::node::NodeType::RELU";
                    case NodeType::AVG_POOL:     return "lantern::cnn::node::NodeType::AVG_POOL";
                    case NodeType::GLOBAL_AVG_POOL:  return "lantern::cnn::node::NodeType::GLOBAL_AVG_POOL";
                    case NodeType::L2_POOL:      return "lantern::cnn::node::NodeType::L2_POOL";
                    case NodeType::FLATTEN:      return "lantern::cnn::node::NodeType::FLATTEN";
                    case NodeType::LEAKY_RELU:      return "lantern::cnn::node::NodeType::LEAKY_RELU";
                    case NodeType::SWISH:      return "lantern::cnn::node::NodeType::SWISH";
                    case NodeType::SIGMOID:      return "lantern::cnn::node::NodeType::SIGMOID";
                    case NodeType::BATCH_NORM:      return "lantern::cnn::node::NodeType::BATCH_NORM";
                    default:                     return "UNKNOWN LANTERN NODE TYPE!";
                }
            }

            /**
             * @brief Get node type from given string
             * @param _str 
             * @return lantern::cnn::node::NodeType
             */
            inline NodeType GetNodeTypeFromString(const std::string& _str) {
                if (_str == "lantern::cnn::node::NodeType::NOTHING")             return NodeType::NOTHING;
                else if (_str == "lantern::cnn::node::NodeType::MAX_POOL")       return NodeType::MAX_POOL;
                else if (_str == "lantern::cnn::node::NodeType::CONVOLVE")      return NodeType::CONVOLVE;
                else if (_str == "lantern::cnn::node::NodeType::RELU")           return NodeType::RELU;
                else if (_str == "lantern::cnn::node::NodeType::AVG_POOL")       return NodeType::AVG_POOL;
                else if (_str == "lantern::cnn::node::NodeType::GLOBAL_AVG_POOL")    return NodeType::GLOBAL_AVG_POOL;
                else if (_str == "lantern::cnn::node::NodeType::L2_POOL")        return NodeType::L2_POOL;
                else if (_str == "lantern::cnn::node::NodeType::FLATTEN")        return NodeType::FLATTEN;
                else if (_str == "lantern::cnn::node::NodeType::LEAKY_RELU")        return NodeType::LEAKY_RELU;
                else if (_str == "lantern::cnn::node::NodeType::SWISH")        return NodeType::SWISH;
                else if (_str == "lantern::cnn::node::NodeType::SIGMOID")        return NodeType::SIGMOID;
                else if (_str == "lantern::cnn::node::NodeType::BATCH_NORM")        return NodeType::BATCH_NORM;
                else                                                        return NodeType::UNKNOWN;
            }

        }
    }
}

namespace nlohmann
{
    template <>
    struct adl_serializer<lantern::cnn::node::NodeType>
    {
        static lantern::cnn::node::NodeType from_json(const json& _j)
        {
            return lantern::cnn::node::GetNodeTypeFromString(_j.get<std::string>());
        }

        static void to_json(json& _j,const lantern::cnn::node::NodeType& _node)
        {
            _j = lantern::cnn::node::GetNodeTypeAsString(_node);
        }
    };
} // namespace nlohmann