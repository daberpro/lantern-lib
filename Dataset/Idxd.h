#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

/**
 * @defgroup LanternIDX IDX Wrapper for lantern
 */

namespace lantern {

    namespace idxd {

        /**
         * @brief IDX wrapper for lantern
         * @tparam T 
         * @ingroup LanternIDX
         */
        template <typename T>
        class Idx {
        private:

            std::ifstream m_file;
            lantern::utility::Vector<int> m_dims;
            lantern::utility::Vector<T> m_data;
            uint8_t m_dimension;
            long long m_total_elements = 1;

            /**
             * @brief Read with big endian for file
             * @param file 
             * @return int
             */
            int ReadBigEndian(std::ifstream& _file){
                uint8_t bytes_[4]; // read 4 bytes_ of _file data
                // take bytes_ from _file because its need char* weh need to cast unsigned char to char
                if (!_file.read(reinterpret_cast<char*>(bytes_), 4)) {
                    throw std::runtime_error("lantern dataset MNIST error, Unexpected EOF while reading big endian data");
                }
                // then left shift all bytes_ to fix order because if we just use implicit cast
                // the order of byte will wrong
                return  (static_cast<int>(bytes_[0]) << 24) |
                        (static_cast<int>(bytes_[1]) << 16) |
                        (static_cast<int>(bytes_[2]) << 8)  |
                        static_cast<int>(bytes_[3]);
            }

        public:

            Idx() noexcept {}

            Idx(const std::string& _file) noexcept {
                try{
                    
                    if(!std::filesystem::exists(_file)){
                        throw std::runtime_error(std::string("Cannot find [")+_file+"]\n");
                    }
                    
                    this->m_file = std::ifstream(_file,std::ios::binary);

                    if(!this->m_file.is_open()){
                        throw std::runtime_error(std::string("Cannot open [")+_file+"]\n");
                    }

                    uint8_t magic_number_[4];
                    this->m_file.read(reinterpret_cast<char*>(magic_number_),4);

                    // check if first two byte is not 0
                    if(magic_number_[0] != 0 || magic_number_[1] != 0){
                        throw std::runtime_error(std::string("File [")+_file+"], is not idx file or invalid format\n");
                    }

                    uint8_t type = magic_number_[2]; // get type
                    this->m_dimension = magic_number_[3]; // get dimension
                    
                    this->m_dims.resizeCapacity(this->m_dimension);
                    for(uint32_t i = 0; i < this->m_dimension; i++){
                        this->m_dims.push_back(
                            this->ReadBigEndian(this->m_file)
                        );
                        this->m_total_elements *= this->m_dims.back();
                    }

                    this->m_data.resizeCapacity(this->m_total_elements);
                    if (std::is_same_v<T,uint8_t> && type == 0x08){
                        this->m_file.read(reinterpret_cast<char*>(this->m_data.data()),this->m_total_elements * sizeof(uint8_t));
                    }else if (std::is_same_v<T,int> && type == 0x0C){
                        for(long long i = 0; i < this->m_total_elements; i++){
                            this->m_data.push_back(
                                this->ReadBigEndian(this->m_file)
                            );
                        }
                    }else {
                        throw std::runtime_error(std::string("File [")+_file+"], invalid format\n");
                    }

                    this->m_file.close();

                }catch(std::runtime_error& error){
                    std::cerr << "Error : " << error.what() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            T* operator [](long long index){
                long long start_ = index;
                if(this->m_dims.size() > 1){
                    for(uint32_t i = 0; i < this->m_dims.size() - 1; i++){
                        start_ *= (this->m_dims.at(i + 1));
                    }
                }
                return this->m_data.data() + start_;
            }
            
            /**
             * @brief Get pointer of data 
             * @return lantern::utility::Vector<T>*
             */
            lantern::utility::Vector<T>* GetData(){
                return &this->m_data;
            }

            /**
             * @brief Get dimension of data
             * @brief - dims 0 for total data
             * @brief - dims 1 for size of width (if exists)
             * @brief - dims 2 for size of height (if exists)
             * @return lantern::utility::Vector<int>*
             */
            lantern::utility::Vector<int>* GetDims(){
                return &this->m_dims;
            }

            /**
             * @brief Get total element for images is :
             * @brief w * h * total_data
             * @brief for labels is :
             * @brief total_data
             * @return long long
             */
            long long GetTotalElements(){
                return this->m_total_elements;
            }

        };


    }

}