#pragma once
#include "../pch.h"
#include <Node.h>
#include <Perceptron.h>

std::ostream &operator<<(std::ostream &os, af::array &tensor)
{
    os << af::toString("tensor",tensor,16,true);
    return os;
}

std::ostream &operator<<(std::ostream &os, latern::Node &node)
{
    os << "Value: " << node.value << ", "
       << "Operator: " << ((int)node.op)
       << "\nGradient: " << node.gradient << " ";
    return os;
}

namespace latern
{
    void print(Node &node)
    {
        std::cout << "\n[ " << std::left << std::setw(3) << node.GetLabel()
                  << "Value: " << std::fixed << std::setprecision(16) << std::setw(3) << node.value
                  << ",Left Child Index: " << std::setw(3) << node.left_child_index
                  << ",Right Child Index: " << std::setw(3) << node.right_child_index
                  << ",Gradient Shape: 1 x " << std::setw(3) << node.total_gradient_size
                  << ",Operator: " << ((int)node.op) << " ]\n"
                  << "Gradient: " << node.gradient;
    }

    void print(perceptron::Perceptron &node)
    {
        std::cout << "\n[ " << std::left << std::setw(3) << node.GetLabel()
                  << "Value: " << std::fixed << std::setprecision(16) << std::setw(3) << node.value
                  << ",Gradient Shape: 1 x " << std::setw(3) << node.total_gradient_size
                  << ",Operator: " << ((int)node.op) << " ]\n"
                  << "Gradient: " << node.gradient;
    }

    template <typename... Args>
    void print(Args&... args)
    {
        std::cout << std::string(70,'=') << "\n";
        ((std::cout << "\n[ " << std::left << std::setw(3) << args.GetLabel()
                    << "Value: " << std::fixed << std::setprecision(16) << std::setw(3) << args.value
                    << ",Gradient Shape: 1 x " << std::setw(3) << args.total_gradient_size
                    << ",Operator: " << ((int)args.op) << " ]\n"
                    << "Gradient: " << args.gradient
                    << "Gradient Based Input: " << args.gradient_based_input),
         ...);
        std::cout << std::string(70,'=') << "\n";
    }

}