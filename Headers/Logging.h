#pragma once
#include "../pch.h"
#include "../AutoGradient/Node.h"
#include "../FeedForwardNetwork/FeedForwardNetwork.h"

std::ostream &operator<<(std::ostream &os,const af::array &tensor)
{
    os << af::toString("tensor",tensor,16,true);
    return os;
}

std::ostream &operator<<(std::ostream &os, lantern::Node &node)
{
    os << "Value: "<< std::fixed << std::setprecision(16) << node.value << ", "
       << "Operator: " << ((int)node.op)
       << "\nGradient: " << node.gradient << " ";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, lantern::utility::Vector<T> &data)
{

    if(std::is_same<double,T>::value){
        for(auto& v : data){
            os << std::fixed << std::setprecision(16) << v << "\n";
        }
    }else{
        os << "\n";
        for(auto& v : data){
            os << v << "\n";
        }
        os << "\n";
    }
    return os;
}
