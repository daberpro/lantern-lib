#pragma once
#include "../pch.h"

namespace latern {

    namespace utility {

        template <typename T>
        class Vector {
        private:
            uint32_t capacity = 0, m_size = 0;
            T* data = nullptr;

            void ResizeCapacity(){
                // set default capacity size when resize
                T* new_container = (T*)::operator new((this->capacity + 10) * sizeof(T));
                uint32_t i = 0;
                for(; i < this->m_size; i++){
                    new(&new_container[i]) T(std::move(this->data[i]));
                    this->data[i].~T();
                }
                ::operator delete(this->data);
                this->data = new_container;
                this->capacity = this->capacity + 10;
            }

        public:

            struct Iterator {

                Iterator(T* ptr) : ptr(ptr){}

                T* ptr = nullptr;
                using category = std::forward_iterator_tag;
                using difference_type = std::ptrdiff_t;

                T& operator *() const {
                    return *this->ptr;
                }

                T* operator ->(){
                    return this->ptr;
                }

                Iterator& operator ++(){
                    this->ptr++;
                    return *this;
                }

                Iterator operator++(int){
                    Iterator tmp = *this;
                    ++(*this);
                    return tmp;
                }

                friend bool operator ==(const Iterator& a, const Iterator& b){
                    return a.ptr == b.ptr;
                }

                friend bool operator !=(const Iterator& a, const Iterator& b){
                    return a.ptr != b.ptr;
                }

            };

            Iterator begin(){
                return Iterator(&this->data[0]);
            }

            Iterator end(){
                return Iterator(&this->data[this->m_size]);
            }

            Vector(uint32_t&& init_capacity): capacity(init_capacity){
                this->ResizeCapacity();
            }

            Vector(std::initializer_list<T> data){
                this->m_size = data.size();
                this->capacity = data.size();
                this->data = (T*)::operator new(this->m_size * sizeof(T));

                uint32_t i = 0;
                for(auto item: data ){
                    new(&this->data[i++]) T(std::move(item));
                }
            }

            Vector(){
                this->ResizeCapacity();
            }

            /**
             * @brief push data into utility vector
             * 
             * @param data 
             */
            void push_back(T&& data){
                if(this->m_size >= this->capacity){
                    this->ResizeCapacity();
                }
                new(&this->data[this->m_size++]) T(std::move(data));
            }

            /**
             * @brief push data into utility vector
             * 
             * @param data 
             */
            void push_back(T& data){
                if(this->m_size >= this->capacity){
                    this->ResizeCapacity();
                }
                new(&this->data[this->m_size++]) T(data);
            }

            /**
             * @brief pop the last item on utility vector
             * 
             */
            void pop_back(){
                if(this->m_size <= 0){
                    std::cerr << "Cannot pop back in latern Vector utility because the size of vector was zero\n";
                    exit(EXIT_FAILURE);
                }
                this->data[--this->m_size].~T();
            }

            /**
             * @brief check if the utility vector was empty
             * 
             * @return true 
             * @return false 
             */
            bool empty(){
                return (this->m_size == 0);
            }

            /**
             * @brief get reference of the last item
             * 
             * @return T& 
             */
            T& back(){
                if(this->m_size <= 0){
                    return this->data[0];
                }
                return this->data[this->m_size-1];
            }

            void setAt(uint32_t&& index, T&& value){
                if(index > this->m_size){
                    std::cerr << "Cannot set utility vector data because index " << index << " is out of bound \n";
                    exit(EXIT_FAILURE);
                }

                new(&this->data[index]) T(std::move(value));
            }

            void setAt(uint32_t& index, T& value){
                if(index > this->m_size){
                    std::cerr << "Cannot set utility vector data because index " << index << " is out of bound \n";
                    exit(EXIT_FAILURE);
                }

                new(&this->data[index]) T(std::move(value));
            }

            bool has(uint32_t&& index){
                return (this->data[index] != nullptr);
            }

            bool has(uint32_t& index){
                return (this->data[index] != nullptr);
            }

            uint32_t size(){
                return this->m_size;
            }

            ~Vector(){
                ::operator delete[](this->data);
            }

            T operator [](uint32_t index){
                if((index < 0 )|| (index > this->m_size)){
                    std::cerr << "Cannot access index " << index << " in latern Vector utility \n";
                    exit(EXIT_FAILURE);
                }
                return this->data[index];
            }

            /**
             * @brief get reference at index, on utility vector
             * 
             * @param index 
             * @return T& 
             */
            T& referenceAt(uint32_t index){
                if((index < 0 )|| (index > this->m_size)){
                    std::cerr << "Cannot access index " << index << " in latern Vector utility \n";
                    exit(EXIT_FAILURE);
                }
                return this->data[index];
            }

            void operator =(std::initializer_list<T> data){
                this->m_size = data.size();
                this->capacity = data.size();
                this->data = (T*)::operator new(this->m_size * sizeof(T));

                uint32_t i = 0;
                for(auto item: data ){
                    new(&this->data[i++]) T(std::move(item));
                }

            }

        };

    }

}