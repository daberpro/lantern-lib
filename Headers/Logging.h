#pragma once
#include "../pch.h"
#include "../AutoGradient/Node.h"
#include "../ConvolutionalNeuralNetwork/Layer.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"

std::ostream &operator<<(std::ostream &os,const lantern::ffn::node::NodeType& type)
{
    os << lantern::ffn::node::GetNodeTypeAsString(type);
    return os;
}

std::ostream &operator<<(std::ostream &os,const af::array &tensor)
{
    os << af::toString("Tensor",tensor,16,true);
    return os;
}

std::ostream &operator<<(std::ostream &os,const lantern::cnn::layer::ConvolveLayerInfo & info)
{
    os << "Kernel Size : " << info.kernel_size << '\n';
    os << "Padding Size : " << info.padding_size << '\n';
    os << "Stride Size : " << info.stride_size << '\n';
    return os;
}

std::ostream &operator<<(std::ostream &os, const lantern::Node &node)
{
    os << "Value: "<< std::fixed << std::setprecision(16) << node.value << ", "
       << "Operator: " << ((int)node.op)
       << "\nGradient: " << node.gradient << " ";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const lantern::utility::Vector<T> & obj)
{

    os << "[";
    for (size_t i = 0; i < obj.size(); ++i) {
        if (i > 0) os << ", ";
        if constexpr (std::is_same_v<T, af::array>) {
            os << af::toString("Tensor", obj[i], 16, true);
        }
        else {
            os << obj[i];
        }
    }
    os << "]";
    return os;
}

template <>
struct std::formatter<af::array> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const af::array& obj, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", af::toString("Tensor",obj,16,true));
    }
};

template <>
struct std::formatter<lantern::utility::Vector<af::array>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const lantern::utility::Vector<af::array>& obj, std::format_context& ctx) const {
        
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < obj.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << af::toString("Tensor", obj[i], 16, true);
        }
        oss << "]";
        return std::format_to(ctx.out(), "{}", oss.str());
    }
};

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