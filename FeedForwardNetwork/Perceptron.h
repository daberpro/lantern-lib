#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {
    
    namespace perceptron {

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
            lantern::utility::Vector<double> vector_velocity;
            lantern::utility::Vector<double> stack_prev_gradient;
            uint32_t layer = 0;
            
            #ifdef OPTIMIZE_VERSION
            Perceptron(double& value, Activation& op): value(value), op(op) {}
            Perceptron(double&& value, Activation&& op): value(value), op(op) {}
            Perceptron(double&& value): value(value) {}
            Perceptron(): value(1.0) {}
            Perceptron(std::string&& label): label(label) {}
            Perceptron(double&& value, std::string&& label): value(value), label(label) {}
            
            double value = 0;
            #endif
            
            #ifdef MATRIX_OPTIMIZE
            Perceptron(double* value, const Activation& op): value(value), op(op) {}
            Perceptron(double* value): value(value) {}
            Perceptron(): value(nullptr) {}
            Perceptron(const std::string& label): label(label) {}
            Perceptron(double* value,const std::string& label): value(value), label(label) {}

            double* value = nullptr;
            #endif

            /**
             * noted the 'prev_child_index' is only use for backpropagation
             * to track prev weight
             * 
             */
            uint32_t total_gradient_size = 0, prev_child_index = 0;
            Activation op = Activation::NOTHING;
            lantern::utility::InitType it = lantern::utility::InitType::XavierGlorot;
            lantern::utility::Vector<Perceptron*> parents;
            lantern::utility::Vector<uint32_t> child_index;

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
