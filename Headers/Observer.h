#pragma once
#include "../pch.h"
#include "Vector.h"

namespace lantern {

    namespace utility {

        
        template <typename... Args>
        class Observer{
        private:
            
            using CallBack = std::function<void(Args...)>;
            lantern::utility::Vector<CallBack> events;

        public:

            Observer(){}

            void Subscribe(CallBack&& callback){
                this->events.emplace_back(std::move(callback));
            }

            void Notify(Args... args){
                for(const auto& event : this->events){
                    event(std::forward<Args>(args)...);
                }
            }

        };

    }

}