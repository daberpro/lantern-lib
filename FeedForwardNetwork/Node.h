#pragma once

namespace lantern {
    namespace ffn {

        namespace node {
    
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
    
            std::string GetNodeTypeAsString(NodeType node) {
                switch(node){
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
    
            NodeType GetNodeTypeFromString(const std::string _str) {
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
