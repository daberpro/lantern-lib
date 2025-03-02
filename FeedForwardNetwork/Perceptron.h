#pragma once
#include "../pch.h"
#include <Vector.h>

namespace latern {
    
    namespace perceptron {

        enum class Activation {
            NOTHING,
            NATURAL_LOG,
            EXP,
            SIN,
            COS,
            TAN,
            SIGMOID,
            RELU,
            SWISH
        };
        
        class Perceptron {
        private:
            bool gradient_init = false, prev_params_init = false, vector_velocity_init = false;
            std::string label = "No-Label";
    
        public:
            /**
             * @brief Construct a new Perceptron object
             * 
             * @param value 
             * @param op 
             */
            Perceptron(double& value, Activation& op):
                value(value), op(op) {}
    
            /**
             * @brief Construct a new Perceptron object
             * 
             * @param value 
             * @param op 
             */
            Perceptron(double&& value, Activation&& op):
                value(value), op(op) {}
    
            /**
             * @brief Construct a new Perceptron object
             * 
             * @param value 
             */
            Perceptron(double&& value): value(value) {}
            
            /**
             * @brief Construct a new Perceptron object
             * 
             * @param value 
             */
            Perceptron(): value(1.0) {}


            /**
             * @brief Construct a new Perceptron object
             * 
             * @param label 
             */
            Perceptron(std::string&& label): label(label) {}
    
            /**
             * @brief Construct a new Perceptron object
             * 
             * @param value 
             * @param label 
             */
            Perceptron(double&& value, std::string&& label): value(value), label(label) {}
            
            double value = 0;
            af::array gradient, gradient_based_input;

            /**
             * NOTED: 
             * this af::array only for SGD-Momentum, AdaptiveGradientDescent only
             */
            af::array vector_velocity;
            af::array stack_prev_gradient;

            /**
             * noted the 'prev_child_index' is only use for backpropagation
             * to track prev weight
             * 
             */
            uint32_t total_gradient_size = 0, prev_child_index = 0;
            Activation op = Activation::NOTHING;
            latern::utility::Vector<Perceptron*> parents;
            latern::utility::Vector<uint32_t> child_index;

            bool IsPrevParamsInit() const;
            void SetPrevParamsInit(bool&& is_init);

            bool IsVectorVelocityInit() const;
            void SetVectorVelocityInit(bool&& is_init);
            /**
             * @brief Return current status of gradient inside node
             * is gradient already initialize or not
             * 
             * @return true 
             * @return false 
             */
            bool IsGradientInit() const;
            /**
             * @brief Set the Gradient Init Perceptron
             * 
             * @param is_init 
             */
            void SetGradientInit(bool&& is_init);
    
            /**
             * @brief Set the Label Perceptron
             * 
             * @param label 
             */
            void SetLabel(std::string&& label);
    
            /**
             * @brief Get the Label Perceptron
             * 
             * @return std::string_view 
             */
            std::string_view GetLabel() const;
            
        };
        
        bool IsIndependentVariable(Perceptron& objective);
    }

}
