#include "Perceptron.h"

namespace lantern
{

    namespace perceptron
    {

        bool IsIndependentVariable(Perceptron &objective)
        {
            return objective.parents.empty();
        };

        bool Perceptron::IsGradientInit() const
        {
            return this->gradient_init;
        };

        void Perceptron::SetGradientInit(bool &&is_init)
        {
            this->gradient_init = is_init;
        };

        void Perceptron::SetLabel(std::string &&label)
        {
            this->label = label;
        };

        std::string_view Perceptron::GetLabel() const
        {
            return this->label;
        };

    }

}

/**
 * Only for logging
 * 
 * @param os 
 * @param op 
 * @return std::ostream& 
 */
std::ostream& operator <<(std::ostream& os, lantern::perceptron::Activation& op){
    
    std::string_view result;
    switch(op){
        case lantern::perceptron::Activation::NOTHING:
            result = "lantern::perceptron::Activation::NOTHING";
        break;
        case lantern::perceptron::Activation::NATURAL_LOG:
            result = "lantern::perceptron::Activation::NATURAL_LOG";
        break;
        case lantern::perceptron::Activation::EXP:
            result = "lantern::perceptron::Activation::EXP";
        break;
        case lantern::perceptron::Activation::SIN:
            result = "lantern::perceptron::Activation::SIN";
        break;
        case lantern::perceptron::Activation::COS:
            result = "lantern::perceptron::Activation::COS";
        break;
        case lantern::perceptron::Activation::TAN:
            result = "lantern::perceptron::Activation::TAN";
        break;
        case lantern::perceptron::Activation::SIGMOID:
            result = "lantern::perceptron::Activation::SIGMOID";
        break;
        case lantern::perceptron::Activation::RELU:
            result = "lantern::perceptron::Activation::RELU";
        break; 
        case lantern::perceptron::Activation::SWISH:
            result = "lantern::perceptron::Activation::SWISH";
        break;
        case lantern::perceptron::Activation::LINEAR:
            result = "lantern::perceptron::Activation::LINEAR";
        break;
        case lantern::perceptron::Activation::TANH:
            result = "lantern::perceptron::Activation::TANH";
        break;
        case lantern::perceptron::Activation::SOFTMAX:
            result = "lantern::perceptron::Activation::SOFTMAX";
        break;
        default:
        result = "--> No Activation Found <--";
    }
    os << result;
    return os;
}