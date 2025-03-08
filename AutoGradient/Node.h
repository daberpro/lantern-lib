#pragma once
#include "../pch.h"

// set floating type to define value for compute
#ifndef FLOATING_TYPE
#define FLOATING_TYPE float
#endif

namespace lantern {
    
    enum class BinOp {
        NOTHING,
        ADDITION,
        SUBSTRACTION,
        MULTIPLY,
        DIVISION,
        NATURAL_LOG,
        EXP,
        SIN,
        COS,
        TAN,
        SIGMOID
    };
    
    class Node {
    private:
        bool gradient_init = false;
        std::string label = "No-Label";

    public:
        /**
         * @brief Construct a new Node object
         * 
         * @param value 
         * @param op 
         */
        Node(FLOATING_TYPE& value, BinOp& op):
            value(value), op(op) {}

        /**
         * @brief Construct a new Node object
         * 
         * @param value 
         * @param op 
         */
        Node(FLOATING_TYPE&& value, BinOp&& op):
            value(value), op(op) {}

        /**
         * @brief Construct a new Node object
         * 
         * @param value 
         */
        Node(const FLOATING_TYPE& value): value(value) {}

        /**
         * @brief Construct a new Node object
         * 
         * @param value 
         * @param label 
         */
        Node(const FLOATING_TYPE& value, std::string&& label): value(value), label(label) {}
        
        FLOATING_TYPE value = 0.0;
        af::array gradient;
        uint32_t total_gradient_size = 0, left_child_index = 0, right_child_index = 0;
        BinOp op = BinOp::NOTHING;
        Node* parents[2] = {};
        

        Node operator *(Node& a);
        Node operator /(Node& a);
        Node operator +(Node& a);
        Node operator -(Node& a);

        /**
         * @brief Return current status of gradient inside node
         * is gradient already initialize or not
         * 
         * @return true 
         * @return false 
         */
        bool IsGradientInit();
        /**
         * @brief Set the Gradient Init Node
         * 
         * @param is_init 
         */
        void SetGradientInit(bool&& is_init);

        /**
         * @brief Set the Label Node
         * 
         * @param label 
         */
        void SetLabel(std::string&& label);

        /**
         * @brief Get the Label Node
         * 
         * @return std::string_view 
         */
        std::string_view GetLabel();
        
    };

    Node Add(Node& a, Node& b);
    Node Substract(Node& a, Node& b);
    Node Division(Node& a, Node& b);
    Node Multiply(Node& a, Node& b);
    Node NaturalLog(Node& a);
    Node Exp(Node& a);
    Node Sin(Node& a);
    Node Cos(Node& a);
    Node Tan(Node& a);
    Node Sigmoid(Node& a);
    
    void CalculateGradient(Node& objective);
    void CalculateGradientO(Node& objective);
    bool IsIndependentVariable(Node& objective);

}
