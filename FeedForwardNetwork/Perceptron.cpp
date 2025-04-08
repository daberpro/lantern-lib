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