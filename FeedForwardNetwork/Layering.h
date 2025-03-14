#pragma once
#include "Perceptron.h"

namespace lantern {

    namespace perceptron {

        class Layer {
        private:
            uint32_t total_layer;
            lantern::utility::Vector<lantern::perceptron::Perceptron*> fix_position_node;

        public:

        /**
         * @brief Get the fix_position_node from layer
         * 
         * @return lantern::utility::Vector<Perceptron*> 
         */
        lantern::utility::Vector<Perceptron*> GetNode(){
            return this->fix_position_node;
        }

        /**
         * @brief Set the Layer
         * 
         * @tparam level 
         * @tparam Args 
         * @param p 
         * @param q 
         */
        template <uint32_t level = 0,typename... Args>
        void SetLayer(Args&... q){
            (
                (
                    q.layer = level,
                    this->fix_position_node.push_back(&q)
                ),
                ...
            );
        }

        };

        template <uint32_t level = 0,typename... Args>
        void SetLayer(Args&... q){
            ((q.layer = level),...);
        }

    }

}