#include "FeedForward.h"

enum class Direction
{
    LEFT,
    RIGHT
};

namespace lantern
{

    void UpdateCalculation(Node *node)
    {
        if (node->parents == nullptr)
        {
            return;
        }

        Node *a = node->parents[0], *b = nullptr;
        if (node->parents[1] != nullptr)
        {
            b = node->parents[1];
        }

        switch (node->op)
        {
        case BinOp::MULTIPLY:
            node->value = a->value * b->value;
            break;
        case BinOp::ADDITION:
            node->value = a->value + b->value;
            break;
        case BinOp::SUBSTRACTION:
            node->value = a->value - b->value;
            break;
        case BinOp::DIVISION:
            node->value = a->value / b->value;
            break;
        case BinOp::NATURAL_LOG:
            node->value = log(a->value);
            break;
        case BinOp::EXP:
            node->value = exp(a->value);
            break;
        case BinOp::SIN:
            node->value = sin(a->value);
            break;
        case BinOp::COS:
            node->value = cos(a->value);
            break;
        case BinOp::TAN:
            node->value = tan(a->value);
            break;
        case BinOp::SIGMOID:
            node->value = 1 / (1 + exp(-a->value));
            break;
        }
    }

    void FeedForward(utility::Vector<Node*>& fix_position_node){
        Node* current_node = nullptr;
        for (int32_t i = fix_position_node.size() - 1; i >= 0;)
        {
            current_node = fix_position_node[i];
            UpdateCalculation(current_node);
            --i;
        }
    }

    void FeedForward(Node *objective, utility::Vector<Node*>& fix_position_node)
    {

        utility::Vector<Node *> all_parents = {objective};
        fix_position_node = {objective};
        Node *current_node = all_parents.back();
        Direction curr_dir = Direction::LEFT;
        while (true)
        {
            if (curr_dir == Direction::LEFT && current_node->parents[0] != nullptr)
            {
                all_parents.push_back(current_node->parents[0]);
                fix_position_node.push_back(current_node->parents[0]);
                current_node = all_parents.back();
            }
            else if (curr_dir == Direction::LEFT)
            {
                curr_dir = Direction::RIGHT;
                current_node = all_parents.back();
                all_parents.pop_back();
            }

            if (curr_dir == Direction::RIGHT && current_node->parents[1] != nullptr)
            {
                all_parents.push_back(current_node->parents[1]);
                fix_position_node.push_back(current_node->parents[1]);
                current_node = all_parents.back();
                curr_dir = Direction::LEFT;
            }
            else if (curr_dir == Direction::RIGHT)
            {
                if (all_parents.empty())
                {
                    break;
                }
                current_node = all_parents.back();
                all_parents.pop_back();
            }
        }

        for (int32_t i = fix_position_node.size() - 1; i >= 0;)
        {
            current_node = fix_position_node[i];
            if (!current_node->IsGradientInit())
            {
                current_node->gradient = af::constant(1.0f, max(current_node->total_gradient_size, 1));
                current_node->SetGradientInit(true);
            }
            UpdateCalculation(current_node);
            --i;
        }
    };

}