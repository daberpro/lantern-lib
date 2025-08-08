#include "Node.h"

namespace lantern
{
    Node Addition(Node &a, Node &b)
    {
        Node n(a.value + b.value, BinOp::ADDITION);
        // set the child index from current left parent gradient size
        n.left_child_index = a.total_gradient_size++;
        n.right_child_index = b.total_gradient_size++;
        n.parents[0] = &a;
        n.parents[1] = &b;
        return std::move(n);
    }

    Node Substract(Node &a, Node &b)
    {
        Node n(a.value - b.value, BinOp::SUBSTRACTION);
        n.left_child_index = a.total_gradient_size++;
        n.right_child_index = b.total_gradient_size++;
        n.parents[0] = &a;
        n.parents[1] = &b;
        return std::move(n);
    }

    Node Multiply(Node &a, Node &b)
    {
        Node n(a.value * b.value, BinOp::MULTIPLY);
        n.left_child_index = a.total_gradient_size++;
        n.right_child_index = b.total_gradient_size++;
        n.parents[0] = &a;
        n.parents[1] = &b;
        return std::move(n);
    }

    Node Division(Node &a, Node &b)
    {
        Node n(a.value / b.value, BinOp::DIVISION);
        n.left_child_index = a.total_gradient_size++;
        n.right_child_index = b.total_gradient_size++;
        n.parents[0] = &a;
        n.parents[1] = &b;
        return std::move(n);
    }

    Node Exp(Node &a)
    {
        Node n(exp(a.value), BinOp::EXP);
        n.left_child_index = a.total_gradient_size++;
        n.parents[0] = &a;
        return std::move(n);
    }

    Node NaturalLog(Node &a)
    {
        Node n(log(a.value), BinOp::NATURAL_LOG);
        n.left_child_index = a.total_gradient_size++;
        n.parents[0] = &a;
        return std::move(n);
    }

    Node Sin(Node &a)
    {
        Node n(sin(a.value), BinOp::SIN);
        n.left_child_index = a.total_gradient_size++;
        n.parents[0] = &a;
        return std::move(n);
    }

    Node Cos(Node &a)
    {
        Node n(cos(a.value), BinOp::COS);
        n.left_child_index = a.total_gradient_size++;
        n.parents[0] = &a;
        return std::move(n);
    }

    Node Tan(Node &a)
    {
        Node n(tan(a.value), BinOp::TAN);
        n.left_child_index = a.total_gradient_size++;
        n.parents[0] = &a;
        return std::move(n);
    }

    Node Sigmoid(Node &a)
    {
        Node n(1 / (1 + exp(-a.value)), BinOp::SIGMOID);
        n.left_child_index = a.total_gradient_size++;
        n.parents[0] = &a;
        return std::move(n);
    }

    Node Node::operator*(Node &a)
    {
        return std::move(Multiply(*(this), a));
    }

    Node Node::operator+(Node &a)
    {
        return std::move(Addition(*(this), a));
    }

    Node Node::operator-(Node &a)
    {
        return std::move(Substract(*(this), a));
    }

    Node Node::operator/(Node &a)
    {
        return std::move(Division(*(this), a));
    }

    bool Node::IsGradientInit()
    {
        return this->gradient_init;
    };
    void Node::SetGradientInit(bool &&is_init)
    {
        this->gradient_init = is_init;
    };

    void Node::SetLabel(std::string &&label)
    {
        this->label = label;
    };

    std::string_view Node::GetLabel()
    {
        return this->label;
    };

    void CalculateGradientO(Node &objective)
    {
        if (!objective.IsGradientInit())
        {
            objective.gradient = af::constant(1.0f, std::max(objective.total_gradient_size, (uint32_t) 1));
            objective.SetGradientInit(true);
        }
        switch (objective.op)
        {
        case BinOp::NATURAL_LOG:
            objective.gradient(0, 0) = objective.gradient(0, 0) / objective.value;
            break;
        case BinOp::EXP:
            objective.gradient(0, 0) = objective.gradient(0, 0) * objective.value;
            break;
        case BinOp::SIN:
            objective.gradient(0, 0) = objective.gradient(0, 0) * cos(objective.value);
            break;
        case BinOp::COS:
            objective.gradient(0, 0) = objective.gradient(0, 0) * -sin(objective.value);
            break;
        case BinOp::TAN:
            objective.gradient(0, 0) = objective.gradient(0, 0) / pow(cos(objective.value), 2);
            break;
        case BinOp::SIGMOID:
            objective.gradient(0, 0) = objective.gradient(0, 0) * (objective.value * (1 - objective.value));
            break;
        };

        if (objective.parents != nullptr)
        {
            CalculateGradient(objective);
        }
    }

    bool IsIndependentVariable(Node &objective)
    {
        return (objective.op == BinOp::NOTHING);
    }

    void CalculateGradient(Node &objective)
    {
        // check if the parent code was nullptr
        // which mean the current Node is an independen variable
        if (objective.parents[0] == nullptr)
        {
            return;
        }

        // declared each the parent node
        // because it using standard form of binary operation like +,-,/,*
        // which mean the parent will be only two in the left handside parent (a)
        // and in the right handside parent (b)
        Node *a = objective.parents[0], *b = nullptr;
        if (!a->IsGradientInit())
        {
            a->gradient = af::constant(1.0f, std::max(a->total_gradient_size, (uint32_t) 1));
            a->SetGradientInit(true);
        }

        // check if the function is single input or not
        if (objective.parents[1] != nullptr)
        {
            b = objective.parents[1];
            if (!b->IsGradientInit())
            {
                b->gradient = af::constant(1.0f, std::max(b->total_gradient_size, (uint32_t) 1));
                b->SetGradientInit(true);
            }
        }

        uint32_t &left_child_index = objective.left_child_index;
        uint32_t &right_child_index = objective.right_child_index;
        switch (objective.op)
        {
        case BinOp::MULTIPLY:
            if(a->total_gradient_size > left_child_index) a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) * b->value;
            if(b->total_gradient_size > right_child_index) b->gradient(right_child_index, 0) = objective.gradient(((right_child_index >= objective.total_gradient_size) ? 0 : right_child_index), 0) * a->value;
            break;
        case BinOp::ADDITION:
            if(a->total_gradient_size > left_child_index) a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0);
            if(b->total_gradient_size > right_child_index) b->gradient(right_child_index, 0) = objective.gradient(((right_child_index >= objective.total_gradient_size) ? 0 : right_child_index), 0);
            break;
        case BinOp::SUBSTRACTION:
            if(a->total_gradient_size > left_child_index) a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0);
            if(b->total_gradient_size > right_child_index) b->gradient(right_child_index, 0) = objective.gradient(((right_child_index >= objective.total_gradient_size) ? 0 : right_child_index), 0);
            break;
        case BinOp::DIVISION:
            if(a->total_gradient_size > left_child_index) a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) / b->value;
            if(a->total_gradient_size > right_child_index) b->gradient(right_child_index, 0) = objective.gradient(((right_child_index >= objective.total_gradient_size) ? 0 : right_child_index), 0) * (-(a->value) / pow(b->value, 2));
            break;
        case BinOp::NATURAL_LOG:
            a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) / a->value;
            break;
        case BinOp::EXP:
            a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) * a->value;
            break;
        case BinOp::SIN:
            a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) * cos(a->value);
            break;
        case BinOp::COS:
            a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) * -sin(a->value);
            break;
        case BinOp::TAN:
            a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) / pow(cos(a->value), 2);
            break;
        case BinOp::SIGMOID:
            a->gradient(left_child_index, 0) = objective.gradient(((left_child_index >= objective.total_gradient_size) ? 0 : left_child_index), 0) * (a->value * (1 - a->value));
            break;
        }
    }

}