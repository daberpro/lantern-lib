#include "Perceptron.h"

namespace latern
{

    namespace perceptron
    {

        bool Perceptron::IsVectorVelocityInit() const{
            return this->vector_velocity_init;
        };

        void Perceptron::SetVectorVelocityInit(bool&& is_init){
            this->vector_velocity_init = is_init;
        };

        bool Perceptron::IsPrevParamsInit() const{
            return this->prev_params_init;
        };

        void Perceptron::SetPrevParamsInit(bool&& is_init){
            this->prev_params_init = is_init;
        };

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