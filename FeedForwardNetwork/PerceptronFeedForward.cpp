#include "PerceptronFeedForward.h"

namespace latern
{

    namespace perceptron
    {

        void PerceptronUpdateCalculation(Perceptron *node)
        {
            if (node->parents.empty() || node->op == Activation::NOTHING)
            {
                return;
            }

            double value = 0.0;
            Perceptron* parent = nullptr;
            uint32_t i = 0;
            for (; i < node->parents.size(); i++)
            {
                parent = node->parents[i];
                value += parent->value * parent->gradient(node->child_index[i],0).scalar<double>();
            }

            /**
             * add bias to the sum product of weight and input
             */
            value += node->gradient((node->total_gradient_size == 0? 1 : node->total_gradient_size),0).scalar<double>();

            switch (node->op)
            {
            case Activation::NATURAL_LOG:
                node->value = log(value);
                break;
            case Activation::EXP:
                node->value = exp(value);
                break;
            case Activation::SIN:
                node->value = sin(value);
                break;
            case Activation::COS:
                node->value = cos(value);
                break;
            case Activation::TAN:
                node->value = tan(value);
                break;
            case Activation::SIGMOID:
                node->value = 1.0 / (1.0 + exp(-value));
                break;
            case Activation::RELU:
                node->value = max(value,0);
                break;
            case Activation::SWISH:
                node->value = value * (1.0 / (1.0 + exp(-value)));
                break;
            }
        }

        void PerceptronFeedForward(utility::Vector<Perceptron *> &fix_position_node)
        {
            Perceptron *current_node = nullptr;
            for (int32_t i = fix_position_node.size() - 1; i >= 0;)
            {
                current_node = fix_position_node[i];
                PerceptronUpdateCalculation(current_node);
                --i;
            }
        }

        void PerceptronFeedForward(Perceptron *objective, utility::Vector<Perceptron *> &fix_position_node)
        {

            if (objective->parents.empty())
            {
                return;
            }

            utility::Vector<Perceptron *> all_parents = {objective}, inputs_node;
            std::unordered_set<Perceptron *> parents_already_added;
            fix_position_node.push_back(objective);
            uint32_t i = 0;
            Perceptron *current_node = nullptr, *parent = nullptr;
            while (!all_parents.empty())
            {
                current_node = all_parents.back();
                all_parents.pop_back();

                i = 0;
                for (; i < current_node->parents.size(); i++)
                {
                    parent = current_node->parents[i];
                    if (parent != nullptr)
                    {
                        all_parents.push_back(parent);
                        if (parents_already_added.find(parent) == parents_already_added.end())
                        {
                            if(!parent->parents.empty()){
                                fix_position_node.push_back(parent);
                            }else{
                                inputs_node.push_back(parent);
                            }
                            parents_already_added.insert(parent);
                        }
                    }
                }
            }

            for(auto v: inputs_node){
                fix_position_node.push_back(v);
            }

            for (int32_t i = fix_position_node.size() - 1; i >= 0;)
            {
                current_node = fix_position_node[i];
                if (!current_node->IsGradientInit())
                {
                    /**
                     * create gradient tensor for current node which will feed
                     * and make sure to add one more slot for bias and check if
                     * current node has op == Activation::NOTHING which mean it was an input
                     * and no need to have bias
                     * after that set current_node->SetGradientInit(true);
                     * 
                     */
                    current_node->gradient = af::randn(max(current_node->total_gradient_size + (current_node->op == Activation::NOTHING? 0 : (current_node->total_gradient_size == 0? 2 : 1)), 1),1,f64);
                    current_node->gradient_based_input = af::constant(1.0,max(current_node->total_gradient_size + (current_node->op == Activation::NOTHING? 0 : (current_node->total_gradient_size == 0? 2 : 1)), 1), f64);
                    current_node->SetGradientInit(true);
                }
                PerceptronUpdateCalculation(current_node);
                --i;
            }

            // set for objective to only has 1 value for gradient
            fix_position_node[0]->gradient(0,0) = 1.0f;

        };


    }

}