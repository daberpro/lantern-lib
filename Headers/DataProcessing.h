#pragma once 
#include "../pch.h"
#include "Vector.h"
#include "File.h"
#include <unordered_set>

/**
 * @defgroup LanternDataProcessing An utility function to manipulate or generate data
 */

namespace lantern {

    namespace data {

        /**
         * @brief Get the Random Sample Class Index
         * 
         * @tparam batch_size 
         * @tparam Args 
         * @param batch_index
         * @param size
         * @ingroup LanternDataProcessing
         */
        template <uint32_t batch_size,typename... Args>
        inline void GetRandomSampleClassIndex(lantern::utility::Vector<uint32_t>& _batch_index,const double& _shuffle = false ,Args... _size){

            _batch_index.clear();
            uint32_t* ptr_ = _batch_index.data();

            std::random_device rd_;
            std::mt19937 rg_(rd_());
            std::uniform_int_distribution<> dis_(0,6); // 6 is just for init, just ignore it
            uint32_t prev_size_ = 0, index_ = 0;
            uint32_t total_class_ = static_cast<uint32_t>(sizeof...(Args));
            uint32_t total_rest_data_ = batch_size % total_class_;
            uint32_t size_each_sample_ = (batch_size - total_rest_data_) / total_class_;

            (([&]()->void{

                dis_ = std::uniform_int_distribution<>(prev_size_,prev_size_ + _size);
                prev_size_ += _size;

                for(uint32_t i = 0; i < size_each_sample_; i++){
                    index_ = dis_(rg_);
                    _batch_index.push_back(index_);
                }

            })(),...);

            dis_ = std::uniform_int_distribution<>(0,prev_size_);
            for(uint32_t i = 0; i < total_rest_data_; i++){
                index_ = dis_(rg_);
                _batch_index.push_back(index_);
            }

            if(_shuffle) std::shuffle(_batch_index.begin(), _batch_index.end(), rg_);

        }

        /**
         * @brief Get the Random Sample Class Index
         *  
         * @tparam Args 
         * @param batch_index
         * @param size
         * @ingroup LanternDataProcessing
         */
        template <typename... Args>
        inline void GetRandomSampleClassIndex(const uint32_t& _batch_size,lantern::utility::Vector<uint32_t>& _batch_index,const double& _shuffle = false ,Args... _size){

            _batch_index.clear();
            uint32_t* ptr_ = _batch_index.data();

            std::random_device rd_;
            std::mt19937 rg_(rd_());
            std::uniform_int_distribution<> dis_(0,6); // 6 is just for init, just ignore it
            uint32_t prev_size_ = 0, index_ = 0;
            uint32_t total_class_ = static_cast<uint32_t>(sizeof...(Args));
            uint32_t total_rest_data_ = _batch_size % total_class_;
            uint32_t size_each_sample_ = (_batch_size - total_rest_data_) / total_class_;

            (([&]()->void{

                dis_ = std::uniform_int_distribution<>(prev_size_,prev_size_ + _size);
                prev_size_ += _size;

                for(uint32_t i = 0; i < size_each_sample_; i++){
                    index_ = dis_(rg_);
                    _batch_index.push_back(index_);
                }

            })(),...);

            dis_ = std::uniform_int_distribution<>(0,prev_size_);
            for(uint32_t i = 0; i < total_rest_data_; i++){
                index_ = dis_(rg_);
                _batch_index.push_back(index_);
            }

            if(_shuffle) std::shuffle(_batch_index.begin(), _batch_index.end(), rg_);

        }

        /**
         * @brief Get the Random Sample Class Index
         * @tparam batch_size 
         * @param batch_index
         * @param _each_size
         * @param _total_size_of_class
         * @ingroup LanternDataProcessing
         */
        template <uint32_t batch_size>
        inline void GetRandomSampleClassIndex(lantern::utility::Vector<uint32_t>& _batch_index,lantern::utility::Vector<uint32_t>& _each_size, const uint32_t& _total_size_of_class,const double& _shuffle = false){
            
            _batch_index.clear();
            _batch_index.resizeCapacity(batch_size);
            std::random_device rd_;
            std::mt19937 rg_(rd_());

            uint32_t* ptr_ = _batch_index.data();
            
            if(_total_size_of_class <= 20){
                for(uint32_t _i = 0; _i < _total_size_of_class; _i++){
                    _batch_index.push_back(_i);
                }

                if(_shuffle) std::shuffle(_batch_index.begin(), _batch_index.end(), rg_);
                return;
            }
            
            std::uniform_int_distribution<> dis_(0,6); // 6 is just for init, just ignore it
            uint32_t prev_size_ = 0, index_ = 0;

            uint32_t total_class_ = _each_size.size();
            uint32_t total_rest_data_ = batch_size % total_class_;
            uint32_t size_each_sample_ = (batch_size - total_rest_data_) / total_class_;

            uint32_t total_size_ = 0;
            for(auto size_ : _each_size){
                dis_ = std::uniform_int_distribution<>(prev_size_,prev_size_ + size_);
                prev_size_ += size_;
                for(uint32_t i = 0; i < size_each_sample_; i++){
                    index_ = dis_(rg_);
                    _batch_index.push_back(index_);
                }
            }

            total_size_ = 0;
            dis_ = std::uniform_int_distribution<>(0, _total_size_of_class);
            while (true) {
                if (total_size_ >= total_rest_data_) {
                    break;
                }
                index_ = dis_(rg_);
                _batch_index.push_back(index_);
                total_size_++;
            }

            if(_shuffle) std::shuffle(_batch_index.begin(), _batch_index.end(), rg_);


        }

        /**
         * @brief Get the Random Sample Class Index
         * @param batch_index
         * @param _each_size
         * @param _total_size_of_class
         * @ingroup LanternDataProcessing
         */
        inline void GetRandomSampleClassIndex(const uint32_t& _batch_size,lantern::utility::Vector<uint32_t>& _batch_index,lantern::utility::Vector<uint32_t>& _each_size, const uint32_t& _total_size_of_class,const double& _shuffle = false){
            
            _batch_index.clear();
            _batch_index.resizeCapacity(_batch_size);
            std::random_device rd_;
            std::mt19937 rg_(rd_());

            uint32_t* ptr_ = _batch_index.data();
            
            if(_total_size_of_class <= 20){
                for(uint32_t _i = 0; _i < _total_size_of_class; _i++){
                    _batch_index.push_back(_i);
                }

                if(_shuffle) std::shuffle(_batch_index.begin(), _batch_index.end(), rg_);
                return;
            }
            
            std::uniform_int_distribution<> dis_(0,6); // 6 is just for init, just ignore it
            uint32_t prev_size_ = 0, index_ = 0;

            uint32_t total_class_ = _each_size.size();
            uint32_t total_rest_data_ = _batch_size % total_class_;
            uint32_t size_each_sample_ = (_batch_size - total_rest_data_) / total_class_;

            uint32_t total_size_ = 0;
            for(auto size_ : _each_size){
                dis_ = std::uniform_int_distribution<>(prev_size_,prev_size_ + size_);
                prev_size_ += size_;
                for(uint32_t i = 0; i < size_each_sample_; i++){
                    index_ = dis_(rg_);
                    _batch_index.push_back(index_);
                }
            }

            total_size_ = 0;
            dis_ = std::uniform_int_distribution<>(0, _total_size_of_class);
            while (true) {
                if (total_size_ >= total_rest_data_) {
                    break;
                }
                index_ = dis_(rg_);
                _batch_index.push_back(index_);
                total_size_++;
            }

            if(_shuffle) std::shuffle(_batch_index.begin(), _batch_index.end(), rg_);


        }


        #define lantern_2d_string_vector lantern::utility::Vector<lantern::string::String>
        std::pair<
            std::span<lantern_2d_string_vector>,
            std::span<lantern_2d_string_vector>
        > PartitionDataset(lantern::file::CSVFile& _csv_file,const float& _train_ratio, const double& _shuffle = false){
            
            if(_train_ratio <= 0){
                throw std::runtime_error("Error PartitionDataset, train ratio or test ratio cannot be negative or zero\n");
            }
            if(_train_ratio > 1){
                throw std::runtime_error("Error PartitionDataset, train ratio or test ratio cannot be greater than 1\n");
            }

            std::random_device rd_;
            std::mt19937 rg_(rd_());
            uint32_t train_size_ = _csv_file.GetRowSize() * _train_ratio - 1;
            uint32_t test_size_ = _csv_file.GetRowSize() - train_size_ - 1;
            auto data_ = _csv_file.GetDataPtr();
            if(_shuffle) std::shuffle(data_->begin(),data_->end(), rg_);
            auto span_ = std::span<lantern_2d_string_vector>(data_->data(), _csv_file.GetRowSize());

            return {
                span_.subspan(0,train_size_),
                span_.subspan(train_size_,test_size_)
            };

        }

    }

}