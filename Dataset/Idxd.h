#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

namespace lantern {

    namespace idxd {

        template <typename T>
        class Idx {
        private:

            std::ifstream file;
            lantern::utility::Vector<int> dims;
            lantern::utility::Vector<T> data;
            uint8_t dimension;
            long long total_elements = 1;


            int ReadBigEndian(std::ifstream& file){
                uint8_t bytes[4]; // read 4 bytes of file data
                // take bytes from file because its need char* weh need to cast unsigned char to char
                file.read(reinterpret_cast<char*>(bytes),4); 
                // then left shift all bytes to fix order because if we just use implicit cast
                // the order of byte will wrong
                return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
            }

        public:

            Idx(){}

            Idx(const std::string& _file){
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
                    
                    this->dims.ResizeCapacity(this->dimension);
                    for(uint32_t i = 0; i < this->dimension; i++){
                        this->dims.push_back(
                            this->ReadBigEndian(this->file)
                        );
                        this->total_elements *= this->dims.back();
                    }

                    this->data.ResizeCapacity(this->total_elements);
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
            
            lantern::utility::Vector<T>* GetData(){
                return &this->data;
            }

            lantern::utility::Vector<int>* GetDims(){
                return &this->dims;
            }

            long long GetTotalElements(){
                return this->total_elements;
            }

        };


    }

}