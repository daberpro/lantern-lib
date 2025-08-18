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

            std::ifstream file;
            lantern::utility::Vector<int> dims;
            lantern::utility::Vector<T> data;
            uint8_t dimension;
            long long total_elements = 1;

            /**
             * @brief Read with big endian for file
             * @param file 
             * @return int
             */
            int ReadBigEndian(std::ifstream& file){
                uint8_t bytes[4]; // read 4 bytes of file data
                // take bytes from file because its need char* weh need to cast unsigned char to char
                if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
                    throw std::runtime_error("lantern dataset MNIST error, Unexpected EOF while reading big endian data");
                }
                // then left shift all bytes to fix order because if we just use implicit cast
                // the order of byte will wrong
                return  (static_cast<int>(bytes[0]) << 24) |
                        (static_cast<int>(bytes[1]) << 16) |
                        (static_cast<int>(bytes[2]) << 8)  |
                        static_cast<int>(bytes[3]);
            }

        public:

            Idx() noexcept {}

            Idx(const std::string& _file) noexcept {
                try{
                    
                    if(!std::filesystem::exists(_file)){
                        throw std::runtime_error(std::string("Cannot find [")+_file+"]\n");
                    }
                    
                    this->file = std::ifstream(_file,std::ios::binary);

                    if(!this->file.is_open()){
                        throw std::runtime_error(std::string("Cannot open [")+_file+"]\n");
                    }

                    uint8_t magic_number[4];
                    this->file.read(reinterpret_cast<char*>(magic_number),4);

                    // check if first two byte is not 0
                    if(magic_number[0] != 0 || magic_number[1] != 0){
                        throw std::runtime_error(std::string("File [")+_file+"], is not idx file or invalid format\n");
                    }

                    uint8_t type = magic_number[2]; // get type
                    this->dimension = magic_number[3]; // get dimension
                    
                    this->dims.resizeCapacity(this->dimension);
                    for(uint32_t i = 0; i < this->dimension; i++){
                        this->dims.push_back(
                            this->ReadBigEndian(this->file)
                        );
                        this->total_elements *= this->dims.back();
                    }

                    this->data.resizeCapacity(this->total_elements);
                    uint32_t loading_bar = 0;
                    if (std::is_same_v<T,uint8_t> && type == 0x08){
                        this->file.read(reinterpret_cast<char*>(this->data.getData()),this->total_elements * sizeof(uint8_t));
                    }else if (std::is_same_v<T,int> && type == 0x0C){
                        for(long long i = 0; i < this->total_elements; i++){
                            this->data.push_back(
                                this->ReadBigEndian(this->file)
                            );
                        }
                    }else {
                        throw std::runtime_error(std::string("File [")+_file+"], invalid format\n");
                    }

                    this->file.close();

                }catch(std::runtime_error& error){
                    std::cerr << "Error : " << error.what() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            T* operator [](long long index){
                long long start = index;
                if(this->dims.size() > 1){
                    for(uint32_t i = 0; i < this->dims.size() - 1; i++){
                        start *= (this->dims.at(i + 1));
                    }
                }
                return this->data.getData() + start;
            }
            
            /**
             * @brief Get pointer of data 
             * @return lantern::utility::Vector<T>*
             */
            lantern::utility::Vector<T>* GetData(){
                return &this->data;
            }

            /**
             * @brief Get dimension of data
             * @brief - dims 0 for total data
             * @brief - dims 1 for size of width (if exists)
             * @brief - dims 2 for size of height (if exists)
             * @return lantern::utility::Vector<int>*
             */
            lantern::utility::Vector<int>* GetDims(){
                return &this->dims;
            }

            /**
             * @brief Get total element for images is :
             * @brief w * h * total_data
             * @brief for labels is :
             * @brief total_data
             * @return long long
             */
            long long GetTotalElements(){
                return this->total_elements;
            }

        };


    }

}