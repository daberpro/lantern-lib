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


namespace lantern
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

    #ifdef OPTIMIZE_VERSION
    void print(perceptron::Perceptron &node)
    {
        std::cout << "\n[ " << std::left << std::setw(3) << node.GetLabel()
                  << "Value: " << std::fixed << std::setprecision(16) << std::setw(3) << node.value
                  << ",Gradient Shape: 1 x " << std::setw(3) << node.total_gradient_size
                  << ",Operator: " << ((int)node.op) << " ]\n"
                  << "Gradient: " << node.gradient;
    }
    #endif

    #ifdef OPTIMIZE_VERSION
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
    #endif

    #ifdef MATRIX_OPTIMIZE

    void print(perceptron::Perceptron &node)
    {
        std::cout << "\n[ " << std::left << std::setw(3) << node.GetLabel()
                  << " Value: " << std::setw(3) << node.value
                  << ",Parents Size: " << std::setw(3) << node.parents.size()
                  << ",Operator: " << ((int)node.op) << " ]\n"
                  << "Gradient: " << node.gradient;
    }

    // void print(lantern::FeedForwardNetwork& ffn){
    //     std::cout << std::string(70,'=') << '\n';
    //     std::cout << " Lantern Feed Forward Network\n";
    //     std::cout << " Input Size : " << ffn.inputs.size() << "\n";
    //     std::cout << " Hidden Size : " << ffn.hiddens.size() << "\n";
    //     std::cout << " Output Size : " << ffn.outputs.size() << "\n";
    //     std::cout << std::string(70,'=') << '\n';
    //     std::cout << "\n Input Layer \n";
    //     for(auto& p : ffn.inputs){
    //         print(p);
    //     }
    //     std::cout << '\n' << std::string(70,'=') << '\n';
    //     std::cout << "\n Hidden Layer \n";
        
    //     for(auto& h : ffn.hiddens){
    //         for(auto& p : h){
    //             print(p);
    //         }
    //     }
    //     std::cout << '\n' << std::string(70,'=') << '\n';
    //     std::cout << "\n Output Layer \n";
    //     for(auto& p : ffn.outputs){
    //         print(p);
    //     }
    //     std::cout << '\n' << std::string(70,'=') << '\n';
    // }
    
    template <typename... Args>
    void print(Args&... args)
    {
        std::cout << std::string(70,'=') << "\n";
        ((std::cout << "\n[ " << std::left << std::setw(3) << args.GetLabel()
                    << "Pointer Value: " << std::fixed << std::setprecision(16) << std::setw(3) << args.value
                    << ",Value: " << std::fixed << std::setprecision(16) << std::setw(3) << (args.value == nullptr? 0 : *(args.value))
                    << ",Gradient Shape: 1 x " << std::setw(3) << args.total_gradient_size
                    << ",Operator: " << ((int)args.op) << " ]\n"
                    << "Gradient: \n" << args.gradient
                    << "\nGradient Based Input: \n" << args.gradient_based_input),
         ...);
        std::cout << std::string(70,'=') << "\n";
    }
    #endif

}