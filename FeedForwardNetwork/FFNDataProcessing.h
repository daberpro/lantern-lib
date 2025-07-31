#pragma once 
#include "../pch.h"
#include "../Headers/Vector.h"
#include <unordered_set>

namespace lantern {

    namespace data {

        /**
         * @brief Get the Random Sample Class Index
         * 
         * @tparam batch_size 
         * @tparam Args 
         * @param batch_index a stack of index data
         * @param size size of each data
         */
        template <uint32_t batch_size,typename... Args>
        void GetRandomSampleClassIndex(lantern::utility::Vector<uint32_t>& batch_index,Args... size){

            batch_index.clear();

            std::random_device rd;
            std::mt19937 rg(rd());
            std::uniform_int_distribution<> dis(0,100);
            uint32_t prev_size = 0, index = 0;
            uint32_t total_class = static_cast<uint32_t>(sizeof...(Args));
            uint32_t total_rest_data = batch_size % total_class;
            uint32_t size_each_sample = (batch_size - total_rest_data) / total_class;
            std::unordered_set<int> already_add;

            (([&]()->void{

                dis = std::uniform_int_distribution<>(prev_size,prev_size + size);
                prev_size += size;

                for(uint32_t i = 0; i < size_each_sample; i++){
                    while(true){
                        index = dis(rg);
                        if(already_add.insert(index).second){
                            batch_index.push_back(index);
                            break;
                        }
                    }
                }

            })(),...);

            dis = std::uniform_int_distribution<>(0,prev_size);
            for(uint32_t i = 0; i < total_rest_data; i++){
                while(true){
                    index = dis(rg);
                    if(already_add.insert(index).second){
                        batch_index.push_back(index);
                        break;
                    }
                }
            }

        }

        /**
         * @brief Get the Random Sample Class Index
         * 
         * @tparam batch_size 
         * @tparam Args 
         * @param batch_index a stack of index data
         * @param size size of each data
         */
        template <uint32_t batch_size>
        void GetRandomSampleClassIndex(lantern::utility::Vector<uint32_t>& batch_index,lantern::utility::Vector<uint32_t>& each_size, const uint32_t& total_size_of_class){
            
            batch_index.clear();
            std::random_device rd;
            std::mt19937 rg(rd());

            
            if(total_size_of_class <= 20){
                lantern::utility::Vector<uint32_t> temp_indexs;
                for(uint32_t _i = 0; _i < total_size_of_class; _i++){
                    temp_indexs.push_back(_i);
                }

                uint32_t* ptr = temp_indexs.getData();
                std::shuffle(&ptr[0],&ptr[temp_indexs.size() - 1],rg);
                batch_index = std::move(temp_indexs);
                return;
            }
            
            std::uniform_int_distribution<> dis(0,100);
            uint32_t prev_size = 0, index = 0;
            std::unordered_set<int> already_add;

            uint32_t total_class = each_size.size();
            uint32_t total_rest_data = batch_size % total_class;
            uint32_t size_each_sample = (batch_size - total_rest_data) / total_class;

            for(auto size : each_size){
                dis = std::uniform_int_distribution<>(prev_size,prev_size + size);
                prev_size += size;
                for(uint32_t i = 0; i < size_each_sample; i++){
                    while(true){
                        index = dis(rg);
                        if(already_add.insert(index).second){
                            batch_index.push_back(index);
                            break;
                        }
                    }
                }
            }

            dis = std::uniform_int_distribution<>(0,prev_size);
            for(uint32_t i = 0; i < total_rest_data; i++){
                while(true){
                    index = dis(rg);
                    if(already_add.insert(index).second){
                        batch_index.push_back(index);
                        break;
                    }
                }
            }

        }

    }

}