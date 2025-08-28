#pragma once
#include "../pch.h"

/**
 * @defgroup LanternNode A node type for lantern
 */

namespace lantern {
    namespace ffn {

        namespace node {
            
            /**
             * @brief Node type of FFN
             * @ingroup LanternNode
             */
            enum class NodeType{
                NOTHING,    // 0
                LINEAR,     // 1
                SIGMOID,    // 2
                RELU,       // 3
                TANH,       // 4
                SWISH,      // 5
                SOFTMAX,    // 6
                UKNOWN
            };
    
            /**
             * @brief Get node as string
             * @param node 
             * @return std::string
             * @ingroup LanternNode
             */
            inline std::string GetNodeTypeAsString(NodeType _node) {
                switch(_node){
                    case NodeType::NOTHING:     return "lantern::ffn::node::NodeType::NOTHING";
                    case NodeType::LINEAR:      return "lantern::ffn::node::NodeType::LINEAR";
                    case NodeType::SIGMOID:     return "lantern::ffn::node::NodeType::SIGMOID";
                    case NodeType::RELU:        return "lantern::ffn::node::NodeType::RELU";
                    case NodeType::TANH:        return "lantern::ffn::node::NodeType::TANH";
                    case NodeType::SWISH:       return "lantern::ffn::node::NodeType::SWISH";
                    case NodeType::SOFTMAX:     return "lantern::ffn::node::NodeType::SOFTMAX";
                    default:                    return "UNKNOWN LANTERN NODE TYPE!";
                }
            }
    
            /**
             * @brief Get enum from string of node
             * @param _str 
             * @return lantern::ffn::node::NodeType
             * @ingroup LanternNode
             */
            inline NodeType GetNodeTypeFromString(const std::string _str) {
                if (_str == "lantern::ffn::node::NodeType::NOTHING")       return NodeType::NOTHING;
                else if (_str == "lantern::ffn::node::NodeType::LINEAR")   return NodeType::LINEAR;
                else if (_str == "lantern::ffn::node::NodeType::SIGMOID")  return NodeType::SIGMOID;
                else if (_str == "lantern::ffn::node::NodeType::RELU")     return NodeType::RELU;
                else if (_str == "lantern::ffn::node::NodeType::TANH")     return NodeType::TANH;
                else if (_str == "lantern::ffn::node::NodeType::SWISH")    return NodeType::SWISH;
                else if (_str == "lantern::ffn::node::NodeType::SOFTMAX")  return NodeType::SOFTMAX;
                else                                                return NodeType::UKNOWN;
            }
            
        }
    }
}



namespace nlohmann
{
    template <>
    struct adl_serializer<lantern::ffn::node::NodeType>
    {
        static lantern::ffn::node::NodeType from_json(const json& _j)
        {
            return lantern::ffn::node::GetNodeTypeFromString(_j);
        }

        static void to_json(json& _j, const lantern::ffn::node::NodeType& _node)
        {
            _j = lantern::ffn::node::GetNodeTypeAsString(_node);
        }
    };
}