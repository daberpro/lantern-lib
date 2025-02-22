#pragma once
#include "pch.h"

std::ostream &operator<<(std::ostream &os, af::array &tensor)
{
    af_print(tensor);
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
                  << "Value: " << std::fixed << std::setprecision(6) << std::setw(3) << node.value
                  << ",Left Child Index: " << std::setw(3) << node.left_child_index
                  << ",Right Child Index: " << std::setw(3) << node.right_child_index
                  << ",Gradient Shape: 1 x " << std::setw(3) << node.total_gradient_size
                  << ",Operator: " << ((int)node.op) << " ]\n"
                  << "Gradient: " << node.gradient;
    }
}