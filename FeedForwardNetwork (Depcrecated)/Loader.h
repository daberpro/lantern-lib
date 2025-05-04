#include "../pch.h"

namespace lantern{

    namespace perceptron {

        /**
         * @brief Show a simple progress bar with current state and total state
         * 
         * @param current 
         * @param total 
         */
        void ProgressBar(int32_t current, int32_t total){

            int32_t width = 70;
            double percentage = static_cast<double>(current) / total;
            int32_t current_pos = static_cast<int32_t>(percentage * width);

            std::cout << "\r[" 
            << std::string(current_pos,'=') 
            << '>'
            << std::string(width - current_pos,' ') 
            << "] "
            << "Progress "
            << static_cast<int32_t>(percentage * 100)
            << "%";

            
        }

    }

}