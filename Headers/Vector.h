#pragma once
#include "../pch.h"

/**
 * @defgroup LanternContainer Custom container implementation for lantern
 */

namespace lantern
{

    namespace utility
    {

        /**
         * @brief Lantern vector utility
         * @tparam T
         * @ingroup LanternContainer
         */
        template <typename T>
        class Vector
        {
        private:
            uint32_t m_capacity = 0, m_size = 0;
            T *m_data = nullptr;

            void CheckIndex(const uint32_t &_index) const
            {
                if (_index >= this->m_size)
                {
                    std::cerr << "Cannot access utility vector data because index " << _index << " is out of bound \n";
                    exit(EXIT_FAILURE);
                }
            }

        public:
            /**
             * @brief Resize the container with new capacity
             * @param _new_capacity
             */
            void resizeCapacity(const uint32_t &_new_capacity)
            {
                // set default capacity size when resize
                T *new_container_ = (T *)::operator new((_new_capacity) * sizeof(T));
                uint32_t i = 0;
                for (; i < this->m_size; i++)
                {
                    new (&new_container_[i]) T(std::move(this->m_data[i]));
                    this->m_data[i].~T();
                }
                if (this->m_data != nullptr)
                {
                    ::operator delete(this->m_data);
                }
                this->m_data = new_container_;
                this->m_capacity = _new_capacity;
            }

            /**
             * @brief Resize container with new capacity and set default value for container
             * @param _new_capacity
             * @param _all_default_value
             */
            void resizeCapacity(const uint32_t &_new_capacity, const T &_all_default_value)
            {
                // set default capacity size when resize
                T *new_container_ = (T *)::operator new((_new_capacity) * sizeof(T));
                uint32_t i = 0;
                for (; i < this->m_size; i++)
                {
                    new (&new_container_[i]) T(std::move(this->m_data[i]));
                    this->m_data[i].~T();
                }
                if (this->m_data != nullptr)
                {
                    ::operator delete(this->m_data);
                }

                this->m_data = new_container_;
                this->m_capacity = _new_capacity;
                // fill the data buffer with default value
                i = 0;
                for (; i < this->m_capacity; i++)
                {
                    this->m_data[i] = _all_default_value;
                }
                this->m_size = this->m_capacity;
            }

            struct Iterator
            {

                explicit Iterator(T *_ptr) : m_ptr(_ptr) {}

                T *m_ptr = nullptr;
                using iterator_category = std::forward_iterator_tag;
                using difference_type = std::ptrdiff_t;
                using value_type = T;
                using pointer = T *;
                using reference = T &;

                reference operator*() const
                {
                    return *this->m_ptr;
                }

                pointer operator->() const
                {
                    return this->m_ptr;
                }

                Iterator &operator++()
                {
                    this->m_ptr++;
                    return *this;
                }

                Iterator operator+(difference_type n) const
                {
                    return Iterator(this->m_ptr + n);
                }

                Iterator operator++(int)
                {
                    Iterator tmp_ = *this;
                    ++(*this);
                    return tmp_;
                }

                friend bool operator==(const Iterator &_a, const Iterator &_b)
                {
                    return _a.m_ptr == _b.m_ptr;
                }

                friend bool operator!=(const Iterator &_a, const Iterator &_b)
                {
                    return _a.m_ptr != _b.m_ptr;
                }
            };

            /**
             * @brief get iterator begin
             *
             * @return Iterator
             */
            Iterator begin()
            {
                return Iterator(&this->m_data[0]);
            }

            /**
             * @brief get end of iterator
             *
             * @return Iterator
             */
            Iterator end()
            {
                return Iterator(&this->m_data[this->m_size]);
            }

            /**
             * @brief Construct a new Vector object
             *
             * @param _init_capacity
             */
            Vector(const uint32_t &_init_capacity)
            {
                this->resizeCapacity(_init_capacity);
            }

            /**
             * @brief Construct a new Vector eith default value
             *
             * @param _init_capacity
             * @param _all_default
             */
            Vector(const uint32_t &_init_capacity, const T &_all_default)
            {
                this->resizeCapacity(_init_capacity, _all_default);
            }

            /**
             * @brief Construct a new Vector object
             *
             * @param _data
             */
            Vector(std::initializer_list<T> _data)
            {
                this->m_size = _data.size();
                this->m_capacity = _data.size();
                this->m_data = (T *)::operator new(this->m_size * sizeof(T));

                uint32_t i = 0;
                for (auto &item_ : _data)
                {
                    new (&this->m_data[i++]) T(std::move(item_));
                }
            }

            /**
             * @brief Construct a new Vector object
             *
             * @param _other
             */
            Vector(const Vector &_other)
            {
                this->m_size = _other.m_size;
                this->m_capacity = _other.m_capacity;
                this->m_data = (T *)::operator new(this->m_capacity * sizeof(T));

                for (uint32_t i = 0; i < this->m_size; i++)
                {
                    new (&this->m_data[i]) T(_other.m_data[i]); // Properly construct objects
                }
            }

            /**
             * @brief Set explicit total item in vector
             * @brief Note: this is not recommended to use, because the vector just allocated space not actual data
             * @param _size
             */
            void explicitTotalItem(const uint32_t &_size)
            {
                if (_size > this->m_capacity)
                {
                    std::cout << "Explicit total item are out of bounds\n";
                    exit(EXIT_FAILURE);
                }
                this->m_size = _size;
            }

            /**
             * @brief Construct a new Vector object
             *
             */
            Vector()
            {
                // because this just init we must set the first size
                this->resizeCapacity(this->m_capacity + 10);
            }

            /**
             * @brief Get the Capacity
             *
             * @return uint32_t
             */
            uint32_t getCapacity() const
            {
                return this->m_capacity;
            }

            /**
             * @brief Emplate back to lantern vector data
             * @tparam ...Args
             * @param ..._data
             */
            template <typename... Args>
            void emplace_back(Args &&..._data)
            {
                if (this->m_size >= this->m_capacity)
                {
                    this->resizeCapacity(this->m_capacity + (this->m_capacity == 0 ? 10 : this->m_capacity / 2));
                }
                new (&this->m_data[this->m_size++]) T(std::forward<Args>(_data)...);
            }

            /**
             * @brief push data into utility vector
             *
             * @param _data
             */
            void push_back(T &&_data)
            {
                if (this->m_size >= this->m_capacity)
                {
                    this->resizeCapacity(this->m_capacity + (this->m_capacity == 0 ? 10 : this->m_capacity / 2));
                }
                new (&this->m_data[this->m_size++]) T(std::move(_data));
            }

            /**
             * @brief push data into utility vector
             *
             * @param _data
             */
            void push_back(const T &_data)
            {
                if (this->m_size >= this->m_capacity)
                {
                    this->resizeCapacity(this->m_capacity + (this->m_capacity == 0 ? 10 : this->m_capacity / 2));
                }
                new (&this->m_data[this->m_size++]) T(_data);
            }

            /**
             * @brief pop the last item on utility vector
             *
             */
            void pop_back()
            {
                if (this->m_size <= 0)
                {
                    std::cerr << "Cannot pop back in lantern Vector utility because the size of vector was zero\n";
                    exit(EXIT_FAILURE);
                }
                this->m_data[this->m_size--].~T();
            }

            /**
             * @brief check if the utility vector was empty
             *
             * @return true
             * @return false
             */
            bool empty() const noexcept
            {
                return (this->m_size == 0);
            }

            /**
             * @brief get reference of the last item
             *
             * @return T&
             */
            T &back()
            {
                if (this->m_data == nullptr)
                {
                    throw std::runtime_error("Runtine Error, cannot access first data at lantern Vector!\n");
                }
                if (this->m_size <= 0)
                {
                    return this->m_data[0];
                }
                return this->m_data[this->m_size - 1];
            }

            /**
             * @brief Get first value in container
             * @return
             */
            T &front()
            {
                if (this->m_data == nullptr)
                {
                    throw std::runtime_error("Runtine Error, cannot access first data at lantern Vector!\n");
                }
                return this->m_data[0];
            }

            /**
             * @brief Get pointer of data at specific index
             *
             * @param _index
             * @return T*
             */
            T *ptrAt(const uint32_t &_index)
            {
                this->CheckIndex(_index);
                return &this->m_data[_index];
            }

            /**
             * @brief Set value at index
             *
             * @param _index
             * @param _value
             */
            void setAt(uint32_t &&_index, T &&_value)
            {
                this->CheckIndex(_index);
                new (&this->m_data[_index]) T(std::move(_value));
            }

            /**
             * @brief Set value at index
             *
             * @param _index
             * @param _value
             */
            void setAt(uint32_t &_index, const T &_value)
            {
                this->CheckIndex(_index);
                new (&this->m_data[_index]) T(std::move(_value));
            }

            /**
             * @brief Check if index was not empty
             *
             * @param index
             * @return true
             * @return false
             */
            bool isIndexEmpty(uint32_t &&_index)
            {
                this->CheckIndex(_index);
                if constexpr (std::is_pointer_v<T>)
                {
                    return this->m_data[_index] != nullptr;
                }
                else
                {
                    return true;
                }
            }

            /**
             * @brief Check if index was no empty
             *
             * @param index
             * @return bool
             */
            bool isIndexEmpty(uint32_t &_index)
            {
                this->CheckIndex(_index);
                if constexpr (std::is_pointer_v<T>)
                {
                    return this->m_data[_index] != nullptr;
                }
                else
                {
                    return true;
                }
            }

            /**
             * @brief Check if the data is in the container
             * @param _data
             * @return bool
             */
            bool has(const T &_data)
            {
                return std::find(this->begin(), this->end(), _data) != this->end();
            }

            /**
             * @brief Get size
             *
             * @return uint32_t
             */
            uint32_t size() const
            {
                return this->m_size;
            }

            ~Vector()
            {
                this->clear();
            }

            /**
             * @brief Get data at index
             *
             * @param _index
             * @return T&
             */
            T &operator[](const uint32_t &_index)
            {
                this->CheckIndex(_index);
                return this->m_data[_index];
            }

            /**
             * @brief Get data at index
             *
             * @param _index
             * @return const T&
             */
            const T &operator[](const uint32_t &_index) const
            {
                this->CheckIndex(_index);
                return this->m_data[_index];
            }

            /**
             * @brief get reference at index, on utility vector
             *
             * @param _index
             * @return T&
             */
            T &referenceAt(uint32_t _index)
            {
                this->CheckIndex(_index);
                return this->m_data[_index];
            }

            void operator=(std::initializer_list<T> _data) noexcept
            {
                this->m_size = _data.size();
                this->m_capacity = _data.size();
                this->m_data = (T *)::operator new(this->m_size * sizeof(T));

                uint32_t i = 0;
                for (auto item_ : _data)
                {
                    new (&this->m_data[i++]) T(std::move(item_));
                }
            }

            Vector &operator=(Vector &&_other) noexcept
            {
                if (this != &_other)
                {
                    // Prevent self-assignment
                    // Free existing memory
                    this->clear();

                    // Transfer ownership
                    this->m_data = _other.m_data;
                    this->m_size = _other.m_size;
                    this->m_capacity = _other.m_capacity;

                    // Leave _other in a valid, destructible state
                    _other.m_data = nullptr;
                    _other.m_size = 0;
                    _other.m_capacity = 0;
                }
                return *this;
            }

            Vector &operator=(Vector &_other) noexcept
            {
                if (this != &_other)
                {
                    // Prevent self-assignment
                    // Free existing memory
                    this->clear();

                    // Copy data
                    this->m_size = _other.m_size;
                    this->m_capacity = _other.m_capacity;
                    this->m_data = (T *)::operator new(this->m_capacity * sizeof(T));
                    for (uint32_t i = 0; i < this->m_size; i++)
                    {
                        new (&this->m_data[i]) T(_other[i]);
                    }
                }

                return *this;
            }

            /**
             * @brief Get the Data
             *
             * @return T*
             */
            T *data() const noexcept
            {
                return this->m_data;
            }

            /**
             * @brief Clear this vector
             *
             */
            void clear()
            {
                // Free existing memory
                if (this->m_data != nullptr)
                {
                    for (uint32_t i = 0; i < this->m_size; i++)
                    {
                        this->m_data[i].~T();
                    }
                    ::operator delete(this->m_data);
                    this->m_size = 0;
                    this->m_capacity = 0;
                    this->m_data = nullptr;
                }
            }

            /**
             * @brief Get data at index
             * @param _index
             * @return T
             */
            T at(const uint32_t &_index)
            {
                this->CheckIndex(_index);
                return this->m_data[_index];
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
         * @ingroup LanternContainer
         */
        template <typename T>
        inline Vector<T> GenerateRandomNormalDVector(const uint32_t &_size, const T &_mean, const T &_stddev)
        {
            std::random_device rd_;
            std::mt19937 gen_(rd_());
            std::normal_distribution<T> dist_(_mean, _stddev);

            Vector<T> result_(_size);
            for (uint32_t i = 0; i < _size; i++)
            {
                result_.push_back(dist_(gen_));
            }
            return result_;
        }

        template <typename T>
        inline Vector<T> GenerateRandomUniformVector(const uint32_t &_size, const T &_mean, const T &_stddev)
        {
            std::random_device rd_;
            std::mt19937 gen_(rd_());
            std::uniform_int<T> dist_(_mean, _stddev);

            Vector<T> result_(_size);
            for (uint32_t i = 0; i < _size; i++)
            {
                result_.push_back(dist_(gen_));
            }
            return result_;
        }

    }

}