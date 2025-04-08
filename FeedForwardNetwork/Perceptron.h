#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {
    
    namespace perceptron {

        /**
         * Actiation function for each node
         * NOTHING mean no operation will do with the node
         */
        enum class Activation {
            NOTHING, // 0
            NATURAL_LOG, // 1
            EXP, // 2
            SIN, // 3
            COS, // 4
            TAN, // 5
            SIGMOID, // 6
            RELU, // 7 
            SWISH, // 8
            LINEAR, // 9
            TANH, // 10
        };
        
        class Perceptron {
        private:

            bool gradient_init = false, prev_params_init = false, vector_velocity_init = false;
            std::string label = "No-Label";
    
        public:

        
            /**
             * NOTED: 
             * this for all optimize except GradientDescent 
             */
            lantern::utility::Vector<double> gradient, gradient_based_input;
            uint32_t layer = 0;
            double value = 0;
            
            Perceptron(double& value, Activation& op): value(value), op(op) {}
            Perceptron(double&& value, Activation&& op): value(value), op(op) {}
            Perceptron(double&& value): value(value) {}
            Perceptron(): value(1.0) {}
            Perceptron(const std::string& label): label(label) {}
            Perceptron(const double& value,const std::string& label): value(value), label(label) {}
            

            /**
             * noted the 'prev_child_index' is only use for backpropagation
             * to track prev weight
             * 
             */
            int32_t total_gradient_size = 0, prev_child_index = 0;
            Activation op = Activation::NOTHING;
            lantern::utility::Vector<Perceptron*> parents;
            lantern::utility::Vector<uint32_t> child_index;

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
