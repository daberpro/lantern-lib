#pragma once
#include "Perceptron.h"

namespace lantern {

    namespace perceptron {

        class Layer {
        private:
            lantern::utility::Vector<lantern::perceptron::Perceptron*> fix_position_node;
            lantern::utility::Vector<uint32_t> total_node_on_layer;

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
             * @brief Get the Total Node On Layer
             * 
             * @param index 
             * @return uint32_t 
             */
            uint32_t GetTotalNodeOnLayer(const uint32_t& index){
                return this->total_node_on_layer[index];
            }

            /**
             * @brief Set the Layer of model
             * 
             * @tparam level 
             * @tparam Args 
             * @param p 
             * @param q 
             */
            template <uint32_t level,typename... Args>
            void SetLayer(Args&... q){
                uint32_t total_node = 0;
                (
                    (
                        q.layer = level,
                        this->fix_position_node.push_back(&q),
                        total_node++
                    ),
                    ...
                );

                this->total_node_on_layer.push_back(std::move(total_node));
            }

            /**
             * @brief Set the Layer with array of perceptron
             * 
             * @param layer 
             */
            void SetLayer(lantern::utility::Vector<lantern::perceptron::Perceptron>& layer, const uint32_t& level = 0){
                uint32_t total_node = 0;
                for(auto& p: layer){
                    p.layer = level;
                    this->fix_position_node.push_back(&p);
                    total_node++;
                }
                this->total_node_on_layer.push_back(std::move(total_node));
            }

        };

    }

}