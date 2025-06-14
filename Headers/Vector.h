#pragma once
#include "../pch.h"

namespace lantern {

    namespace utility {

        enum class InitType {
            XavierGlorot,
            HeKaiming
        };

        template <typename T>
        class Vector {
        private:
            uint32_t capacity = 0, m_size = 0;
            T* data = nullptr;

            void ResizeCapacity(const uint32_t& new_capacity){
                // set default capacity size when resize
                T* new_container = (T*)::operator new((new_capacity) * sizeof(T));
                uint32_t i = 0;
                for(; i < this->m_size; i++){
                    new(&new_container[i]) T(std::move(this->data[i]));
                    this->data[i].~T();
                }
                ::operator delete(this->data);
                this->data = new_container;
                this->capacity = new_capacity;
            }

            void ResizeCapacity(const uint32_t& new_capacity,const T& all_default_value){
                // set default capacity size when resize
                T* new_container = (T*)::operator new((new_capacity) * sizeof(T));
                uint32_t i = 0;
                this->data = new_container;
                this->capacity = new_capacity;
                // fill the data buffer with default value
                i = 0;
                for(; i < this->capacity; i++){
                    this->data[i] = all_default_value;
                }
                this->m_size = this->capacity;
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

            /**
             * @brief get iterator begin 
             * 
             * @return Iterator 
             */
            Iterator begin(){
                return Iterator(&this->data[0]);
            }

            /**
             * @brief get end of iterator
             * 
             * @return Iterator 
             */
            Iterator end(){
                return Iterator(&this->data[this->m_size]);
            }

            /**
             * @brief Construct a new Vector object
             * 
             * @param init_capacity 
             */
            Vector(const uint32_t& init_capacity): capacity(init_capacity){
                this->ResizeCapacity(this->capacity);
            }

            /**
             * @brief Construct a new Vector eith default value
             * 
             * @param init_capacity 
             * @param all_default 
             */
            Vector(const uint32_t& init_capacity,const T& all_default): capacity(init_capacity){
                this->ResizeCapacity(this->capacity, all_default);
            }

            /**
             * @brief Construct a new Vector object
             * 
             * @param data 
             */
            Vector(std::initializer_list<T> data){
                this->m_size = data.size();
                this->capacity = data.size();
                this->data = (T*)::operator new(this->m_size * sizeof(T));

                uint32_t i = 0;
                for(auto item: data ){
                    new(&this->data[i++]) T(std::move(item));
                }
            }

            /**
             * @brief Construct a new Vector object
             * 
             * @param other 
             */
            Vector(const Vector& other) {
                this->m_size = other.m_size;
                this->capacity = other.capacity;
                this->data = (T*)::operator new(this->capacity * sizeof(T));
            
                for (uint32_t i = 0; i < this->m_size; i++) {
                    new(&this->data[i]) T(other.data[i]); // Properly construct objects
                }
            }
            
            void explicitTotalItem(const uint32_t& size){
                if(size > this->capacity || size < 0){
                    std::cout << "Explicit total item are out of bounds\n";
                    exit(EXIT_FAILURE);
                }
                this->m_size = size;
            }

            /**
             * @brief Construct a new Vector object
             * 
             */
            Vector(){
                this->ResizeCapacity(this->capacity + 10);
            }

            /**
             * @brief Get the Capacity 
             * 
             * @return uint32_t 
             */
            uint32_t getCapacity() const {
                return this->capacity;
            }

            /**
             * @brief push data into utility vector
             * 
             * @param data 
             */
            void push_back(T&& data){
                if(this->m_size >= this->capacity){
                    this->ResizeCapacity(this->capacity + 10);
                }
                new(&this->data[this->m_size++]) T(std::move(data));
            }

            /**
             * @brief push data into utility vector
             * 
             * @param data 
             */
            void push_back(const T& data){
                if(this->m_size >= this->capacity){
                    this->ResizeCapacity(this->capacity + 10);
                }
                new(&this->data[this->m_size++]) T(data);
            }

            /**
             * @brief pop the last item on utility vector
             * 
             */
            void pop_back(){
                if(this->m_size <= 0){
                    std::cerr << "Cannot pop back in lantern Vector utility because the size of vector was zero\n";
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

            /**
             * @brief Get pointer of data at specific index
             * 
             * @param index 
             * @return T* 
             */
            T* ptrAt(const uint32_t& index){
                if(index > this->m_size){
                    std::cerr << "Cannot get utility vector pointer because index " << index << " is out of bound \n";
                    exit(EXIT_FAILURE);
                }

                return &this->data[index];
            }

            /**
             * @brief Set value at index
             * 
             * @param index 
             * @param value 
             */
            void setAt(uint32_t&& index, T&& value){
                if(index > this->m_size){
                    std::cerr << "Cannot set utility vector data because index " << index << " is out of bound \n";
                    exit(EXIT_FAILURE);
                }

                new(&this->data[index]) T(std::move(value));
            }

            /**
             * @brief Set value at index
             * 
             * @param index 
             * @param value 
             */
            void setAt(uint32_t& index, const T& value){
                if(index > this->m_size){
                    std::cerr << "Cannot set utility vector data because index " << index << " is out of bound \n";
                    exit(EXIT_FAILURE);
                }

                new(&this->data[index]) T(std::move(value));
            }

            /**
             * @brief Check if index was not empty
             * 
             * @param index 
             * @return true 
             * @return false 
             */
            bool has(uint32_t&& index){
                return (this->data[index] != nullptr);
            }

            /**
             * @brief Check if index was no empty
             * 
             * @param index 
             * @return true 
             * @return false 
             */
            bool has(uint32_t& index){
                return (this->data[index] != nullptr);
            }

            /**
             * @brief Get size
             * 
             * @return uint32_t 
             */
            uint32_t size(){
                return this->m_size;
            }

            ~Vector(){
                this->clear();
            }

            /**
             * @brief Get data at index
             * 
             * @param index 
             * @return const T& 
             */
            const T& operator [](const uint32_t& index) const{
                if((index < 0 )|| (index > this->m_size)){
                    std::cerr << "Cannot access index " << index << " in lantern Vector utility \n";
                    __debugbreak();
                    exit(EXIT_FAILURE);
                }
                return this->data[index];
            }


            /**
             * @brief Get data at index
             * 
             * @param index 
             * @return T& 
             */
            T& operator [](const uint32_t& index) {
                if((index < 0 )|| (index > this->m_size)){
                    std::cerr << "Cannot access index " << index << " in lantern Vector utility \n";
                    __debugbreak();
                    exit(EXIT_FAILURE);
                }
                return this->data[index];
            }

            /**
             * @brief Get data at index
             * 
             * @param index 
             * @return const T& 
             */
            const T& operator [](const int32_t& index) const{
                if((index < 0 )|| (index > this->m_size)){
                    std::cerr << "Cannot access index " << index << " in lantern Vector utility \n";
                    __debugbreak();
                    exit(EXIT_FAILURE);
                }
                return this->data[index];
            }

            /**
             * @brief Get data at index
             * 
             * @param index 
             * @return T& 
             */
            T& operator [](const int32_t& index) {
                if((index < 0 )|| (index > this->m_size)){
                    std::cerr << "Cannot access index " << index << " in lantern Vector utility \n";
                    __debugbreak();
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
                    std::cerr << "Cannot access index " << index << " in lantern Vector utility \n";
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

            void operator =(Vector&& other){
                if (this != &other) { // Prevent self-assignment
                    // Free existing memory
                    for (uint32_t i = 0; i < this->m_size; i++) {
                        this->data[i].~T();
                    }
                    ::operator delete(this->data);
            
                    // Transfer ownership
                    this->data = other.data;
                    this->m_size = other.m_size;
                    this->capacity = other.capacity;
            
                    // Leave `other` in a valid state
                    other.data = nullptr;
                    other.m_size = 0;
                    other.capacity = 0;
                }
            }

            /**
             * @brief Pointer other vector to this vector
             * 
             * @param based 
             */
            void setPtrData(Vector& based){
                this->data = based.getData();
                this->m_size = based.size();
                this->capacity = based.getCapacity();
            }

            /**
             * @brief Copy pointer data to this vector
             * 
             * @param based 
             */
            void copyPtrData(Vector& based){
                this->clear();
                this->m_size = based.size();
                this->capacity = based.getCapacity();

                this->data = (T*)::operator new(this->capacity * sizeof(T));
                for(uint32_t i = 0; i < this->m_size; i++){
                    new(&this->data[i]) T(based.getData()[i]);
                }
            }

            /**
             * @brief Get the Data
             * 
             * @return T* 
             */
            T* getData(){
                return this->data;
            }

            /**
             * @brief Clear this vector
             * 
             */
            void clear(){
                // Free existing memory
                for (uint32_t i = 0; i < this->m_size; i++) {
                    this->data[i].~T();
                }
                ::operator delete(this->data);
                
                this->m_size = 0;
                this->capacity = 0;
                this->data = nullptr;
            }

        };

        /**
         * @brief Generate rnadom normal distribution vector
         * 
         * @tparam T 
         * @param size 
         * @param mean 
         * @param stddev 
         * @return Vector<T> 
         */
        template <typename T>
        Vector<T> GenerateRandomNormalDVector(const uint32_t& size, const T& mean, const T& stddev) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(mean, stddev);

            Vector<T> result(size);
            for (uint32_t i = 0; i < size; i++) {
                result.push_back(dist(gen));
            }
            return result;
        }

    }

}