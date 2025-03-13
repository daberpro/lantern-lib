#pragma once
#include "Perceptron.h"

namespace lantern {

    namespace perceptron {

        class Layer {
        private:
            uint32_t total_layer;
            lantern::utility::Vector<Perceptron*> fix_position_node;
            #ifdef MATRIX_OPTIMIZE
            af::array parameters, param;
            #endif

        public:

        lantern::utility::Vector<Perceptron*> GetNode(){
            return this->fix_position_node;
        }

        template <uint32_t level = 0,typename... Args>
        void SetLayer(Args&... q){
            ((q.layer = level),...);
        }

        };

        template <uint32_t level = 0,typename... Args>
        void SetLayer(Args&... q){
            ((q.layer = level),...);
        }

    }

}