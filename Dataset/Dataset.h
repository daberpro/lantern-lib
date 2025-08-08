#pragma once
#include "../pch.h"
#include "Idxd.h"

namespace lantern {

    namespace dataset {

        class MnistDataset {
        private:

            std::string path, current_path;
            lantern::idxd::Idx<uint8_t> train_images;
            lantern::idxd::Idx<uint8_t> train_labels;
            lantern::idxd::Idx<uint8_t> images;
            lantern::idxd::Idx<uint8_t> labels;

        public:

            MnistDataset() {
                this->current_path = std::filesystem::current_path().string();
                std::string dataset_dir = this->current_path + "/dataset/";
                this->train_images = lantern::idxd::Idx<uint8_t>(dataset_dir + "train-images.idx3-ubyte");
                this->train_labels = lantern::idxd::Idx<uint8_t>(dataset_dir + "train-labels.idx1-ubyte");
                this->images = lantern::idxd::Idx<uint8_t>(dataset_dir + "t10k-images.idx3-ubyte");
                this->labels = lantern::idxd::Idx<uint8_t>(dataset_dir + "t10k-labels.idx1-ubyte");
            }


            void PrintTrainDataAt(const uint32_t& index){
                auto image = this->train_images[static_cast<long long>(index)];
                auto* dims = this->train_images.GetDims();
                for(uint32_t row = 0; row < dims->at(1); row++){
                    for(uint32_t col = 0; col < dims->at(2); col++){
                        std::cout << (image[row * dims->at(1) + col] > 122? "#" : ".") << " ";
                    }
                    std::cout << '\n';
                }
                std::cout << "Label : " << static_cast<int>(*this->train_labels[index]) << '\n';
            }

            uint8_t* GetTrainImageAt(long long index){
               return this->train_images[index];
            }

            uint8_t* GetTrainLabelAt(long long index){
               return this->train_images[index];
            }

            uint8_t* GetImageAt(long long index){
               return this->images[index];
            }

            uint8_t* GetLabelAt(long long index){
               return this->images[index];
            }

            lantern::utility::Vector<int>* GetTrainImageDims(){
                return this->train_images.GetDims();
            }

            lantern::utility::Vector<int>* GetTrainLabelDims(){
                return this->train_labels.GetDims();
            }

            lantern::utility::Vector<int>* GetImageDims(){
                return this->images.GetDims();
            }
            
            lantern::utility::Vector<int>* GetLabelDims(){
                return this->train_labels.GetDims();
            }

            std::string_view GetPath(){
                return this->path;
            }
        
        };

    }

}