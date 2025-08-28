#pragma once
#include "../pch.h"
#include "Vector.h"
#include "Utility.h"
#include "LanternString.h"
#include <H5Cpp.h>

/**
 * @defgroup LanternFile File manager for lantern
 */

namespace lantern {

    namespace file {

        /**
         * @brief Lantern CSV file wrapper
         * @ingroup LanternFile
         */
        class CSVFile {
        private:
            lantern::utility::Vector<lantern::utility::Vector<lantern::string::String>> m_data;
            std::unordered_map<std::string, uint32_t> m_index_map;
            uint32_t m_index = 0;

            void CheckFileEmpty() const {
                if(this->m_data.empty()){
                    throw std::runtime_error(std::format("Error CSVFile, file is empty"));
                }
            }

        public:

            CSVFile(){}
            CSVFile(CSVFile&& _file) noexcept {
                this->m_data = std::move(_file.m_data);
            }

            void operator =(CSVFile&& _file) noexcept {
                this->m_data = std::move(_file.m_data);
            }
            /**
             * @brief Get pointer to data inside CSV file
             * @return lantern::utility::Vector<lantern::utility::Vector<std::string>>
             */
            auto* GetDataPtr() {
                return &this->m_data;
            }

            /**
             * @brief Get pointer of index map
             * @return std::unordered_map<std::string, uint32_t>*
             */
            auto* GetIndexMapPtr() {
                return &this->m_index_map;
            }

            /**
             * @brief Get data in specific row, col and cast to type T
             * @tparam T 
             * @param row 
             * @param col 
             * @return T
             */
            template <typename T>
            T Get(const uint32_t& _row, const uint32_t _col) {
                this->CheckFileEmpty();
                if (_col >= this->m_data.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access column index \"{}\" out of bound", _col));
                }
                if (_row >= this->m_data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _row));
                }
                return this->m_data[_row][_col].as<T>();
            }

            // ====================================================================================
            // COLUMN DEFINITION
            // ====================================================================================
            /**
             * @brief Get column at index and cast to T type
             * @tparam T 
             * @param _index 
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Col(const uint32_t& _index) {
                this->CheckFileEmpty();
                if (_index >= this->m_data.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access column index \"{}\" out of bound", _index));
                }
                lantern::utility::Vector<T> result_(this->m_data.size());
                result_.explicitTotalItem(this->m_data.size());
                std::transform(
                    this->m_data.begin(),
                    this->m_data.end(),
                    result_.begin(),
                    [&](const lantern::utility::Vector<lantern::string::String>& _row) -> T {
                        return _row[_index].as<T>();
                    }
                );
                return result_;
            }

            template <typename T>
            auto Col(const uint32_t& _index, const uint32_t& _start_index, const uint32_t& _count) {
                this->CheckFileEmpty();
                if (_index >= this->m_data.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access column index \"{}\" out of bound", _index));
                }
                
                lantern::utility::Vector<lantern::string::String> holder_(this->m_data.size());
                for(auto& row_ : this->m_data){
                    holder_.push_back(row_[_index]);
                }

                auto result_ = lantern::utility::Vector<T>(_count);
                auto span_ = std::span<lantern::string::String>(holder_.data(),holder_.size());
                auto subspan_ = span_.subspan(_start_index,_count);
                if constexpr (std::is_same_v<T,std::string> || std::is_same_v<T,lantern::string::String>){
                    for(uint32_t i = 0; i < subspan_.size(); i++){
                        result_.push_back(
                            static_cast<T>(subspan_[i])
                        );
                    }
                    return result_;
                }
                
                result_.explicitTotalItem(_count);
                std::transform(
                    subspan_.begin(),
                    subspan_.end(),
                    result_.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );

                return result_;
            }

            // ====================================================================================

            // ====================================================================================
            // ROW DEFINITION
            // ====================================================================================
            /**
             * @brief Get row at index and cast to T type
             * @tparam T
             * @param _index
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Row(const uint32_t& _index) {
                this->CheckFileEmpty();
                if (_index >= this->m_data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }
                
                auto& data_ = this->m_data[_index];
                uint32_t allocated_size_ = data_.size();
                lantern::utility::Vector<T> result_(allocated_size_);
                
                if constexpr (std::is_same_v<T,std::string> || std::is_same_v<T,lantern::string::String>){
                    for(uint32_t i = 0; i < allocated_size_; i++){
                        result_.push_back(
                            static_cast<T>(data_[i])
                        );
                    }
                    return result_;
                }
                
                if(result_.size() < data_.size()) result_.explicitTotalItem(allocated_size_);
                std::transform(
                    data_.begin(),
                    data_.end(),
                    result_.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );
                return result_;
            }

            /**
             * @brief Get row at index and cast to T type, and store into utility vector
             * @tparam T
             * @param _target_out
             * @param _index
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Row(lantern::utility::Vector<T>& _target_out,const uint32_t& _index) {
                this->CheckFileEmpty();
                if (_index >= this->m_data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }
                
                auto& data_ = this->m_data[_index];
                uint32_t allocated_size_ = data_.size();
                if(_target_out.size() < allocated_size_){
                    _target_out.resizeCapacity(allocated_size_);
                    _target_out.explicitTotalItem(allocated_size_);
                }
                
                if constexpr (std::is_same_v<T,std::string> || std::is_same_v<T,lantern::string::String>){
                    for(uint32_t i = 0; i < allocated_size_; i++){
                        new(_target_out.ptrAt(i)) T(static_cast<T>(data_[i]));
                    }
                    return;
                }
                
                std::transform(
                    data_.begin(),
                    data_.end(),
                    _target_out.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );
            }
            
            /**
             * @brief Get row at index, and get columns of row as define range
             * @tparam T
             * @param _index
             * @param _index_start
             * @param _count 
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Row(const uint32_t _index,const uint32_t& _index_start,const uint32_t& _count) {
                this->CheckFileEmpty();
                if (_index_start + _count >= this->m_data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index_start + _count));
                }

                auto result_ = lantern::utility::Vector<T>(_count);
                auto& data_ = this->m_data[_index];
                auto span_ = std::span<lantern::string::String>(data_.data(), data_.size());
                auto subspan_ = span_.subspan(_index_start,_count);

                if constexpr (std::is_same_v<T,std::string> || std::is_same_v<T,lantern::string::String>){
                    for(uint32_t i = 0; i < subspan_.size(); i++){
                        result_.push_back(
                            static_cast<T>(subspan_[i])
                        );
                    }
                    return result_;
                }

                // this only need if we not use push_back or emplace_back
                result_.explicitTotalItem(_count);
                std::transform(
                    subspan_.begin(),
                    subspan_.end(),
                    result_.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );

                return result_;
            }

            /**
             * @brief Get row at index of span, and get columns of row as define range
             * @tparam T
             * @param _span
             * @param _index
             * @param _index_start
             * @param _count 
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Row(const std::span<lantern::utility::Vector<lantern::string::String>>& _span,const uint32_t _index,const uint32_t& _index_start,const uint32_t& _count) {
                if(_span.empty()){
                    throw std::runtime_error("Error CSVFile, span are empty");
                }
                if(_index > _span.size()){
                    throw std::runtime_error(std::format("Error CSVFile, row of span index \"{}\", out of bound", _index));
                }
                if (_index_start + _count > _span.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access range column \"{}\" of row index \"{}\", out of bound", _index_start + _count, _index));
                }

                auto result_ = lantern::utility::Vector<T>(_count);
                auto& data_ = _span[_index];
                auto span_ = std::span<lantern::string::String>(data_.data(), data_.size());
                auto subspan_ = span_.subspan(_index_start,_count);

                if constexpr (std::is_same_v<T,std::string> || std::is_same_v<T,lantern::string::String>){
                    for(uint32_t i = 0; i < subspan_.size(); i++){
                        result_.push_back(static_cast<T>(subspan_[i]));
                    }
                    return result_;
                }

                // this only need if we not use push_back or emplace_back
                if(result_ < subspan_.size()) result_.explicitTotalItem(_count);
                std::transform(
                    subspan_.begin(),
                    subspan_.end(),
                    result_.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );

                return result_;
            }

            /**
             * @brief Get row at index of span, and get columns of row as define range and store it into utility vector
             * @tparam T
             * @param _target_out
             * @param _span
             * @param _index
             * @param _start_index
             * @param _count 
             */
            template <typename T>
            void Row(lantern::utility::Vector<T>& _target_out,const std::span<lantern::utility::Vector<lantern::string::String>>& _span,const uint32_t _index,const uint32_t& _index_start,const uint32_t& _count) {
                if(_span.empty()){
                    throw std::runtime_error("Error CSVFile, span are empty");
                }
                if(_index > _span.size()){
                    throw std::runtime_error(std::format("Error CSVFile, row of span index \"{}\", out of bound", _index));
                }
                if (_index_start + _count > _span.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access range column \"{}\" of row index \"{}\", out of bound", _index_start + _count, _index));
                }

                auto& data_ = _span[_index];
                auto span_ = std::span<lantern::string::String>(data_.data(), data_.size());
                auto subspan_ = span_.subspan(_index_start,_count);

                // this only need if we not use push_back or emplace_back
                if(_target_out.size() < subspan_.size()){
                    _target_out.resizeCapacity(_count);
                    _target_out.explicitTotalItem(_count);
                }

                if constexpr (std::is_same_v<T,std::string> || std::is_same_v<T,lantern::string::String>){
                    for(uint32_t i = 0; i < subspan_.size(); i++){
                        new(_target_out.ptrAt(i)) T(static_cast<T>(subspan_[i]));
                    }
                    return;
                }

                std::transform(
                    subspan_.begin(),
                    subspan_.end(),
                    _target_out.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );
            }

            /**
             * @brief Get row of span at index
             * @tparam T
             * @param _span
             * @param _index
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Row(const std::span<lantern::utility::Vector<lantern::string::String>>& _span,const uint32_t& _index) {
                if(_span.empty()){
                    throw std::runtime_error("Error CSVFile, span are empty");
                }
                if(_index > _span.size()){
                    throw std::runtime_error(std::format("Error CSVFile, row of span index \"{}\", out of bound", _index));
                }

                auto& data_ = _span[_index];
                uint32_t allocated_size_ = data_.size();
                lantern::utility::Vector<T> result_(allocated_size_);
                result_.explicitTotalItem(allocated_size_);
                std::transform(
                    data_.begin(),
                    data_.end(),
                    result_.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );
                return result_;
            }

            /**
             * @brief Get row of span at index, and store it into utility vector
             * @tparam T
             * @param _target_out
             * @param _span
             * @param _index
             */
            template <typename T>
            void Row(lantern::utility::Vector<T>& _target_out,const std::span<lantern::utility::Vector<lantern::string::String>>& _span,const uint32_t& _index) {
                if(_span.empty()){
                    throw std::runtime_error("Error CSVFile, span are empty");
                }
                if(_index > _span.size()){
                    throw std::runtime_error(std::format("Error CSVFile, row of span index \"{}\", out of bound", _index));
                }

                auto& data_ = _span[_index];
                uint32_t allocated_size_ = data_.size();
                if(_target_out.size() < allocated_size_){
                    _target_out.resizeCapacity(allocated_size_);
                    _target_out.explicitTotalItem(allocated_size_);
                }
                std::transform(
                    data_.begin(),
                    data_.end(),
                    _target_out.begin(),
                    [&](const lantern::string::String& _str) -> T {
                        return _str.as<T>();
                    }
                );
            }

            // ====================================================================================

           
            /**
             * @brief Get specific value string in row, and mapping it into numeric 
             * @param _index 
             * @param _index_start 
             * @param _count 
             * @return lantern::utility::Vector<uint32_t>
             */
            auto RowIndexMapping(const uint32_t _index,const uint32_t& _index_start,const uint32_t& _count) {
                this->CheckFileEmpty();
                if (_index_start + _count >= this->m_data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }

                auto result_ = lantern::utility::Vector<uint32_t>(_count);
                auto& data_ = this->m_data[_index];
                auto span_ = std::span<lantern::string::String>(data_.data(), data_.size());
                auto subspan_ = span_.subspan(_index_start,_count);

                for(uint32_t i = 0; i < subspan_.size(); i++){
                    auto str_ = static_cast<const std::string>(subspan_[i]);
                    if(this->m_index_map.contains(str_)){
                        result_.push_back(this->m_index_map[str_]);
                    }else{
                        this->m_index_map.insert({
                            str_,
                            this->m_index++
                        });
                        result_.push_back(this->m_index_map[str_]);
                    };   
                }
                return result_;
            }

            void RowIndexMapping(lantern::utility::Vector<uint32_t>& _target_out,const std::span<lantern::utility::Vector<lantern::string::String>>& _span,const uint32_t _index,const uint32_t& _index_start,const uint32_t& _count) {
                if (_index_start + _count >= _span.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }

                auto& data_ = _span[_index];
                auto span_ = std::span<lantern::string::String>(data_.data(), data_.size());
                auto subspan_ = span_.subspan(_index_start,_count);

                if(_target_out.size() < subspan_.size()){
                    _target_out.resizeCapacity(subspan_.size());
                    _target_out.explicitTotalItem(subspan_.size());
                }
                for(uint32_t i = 0; i < subspan_.size(); i++){
                    auto str_ = static_cast<const std::string>(subspan_[i]);
                    if(this->m_index_map.contains(str_)){
                        new(_target_out.ptrAt(i)) uint32_t(this->m_index_map[str_]);
                    }else{
                        this->m_index_map.insert({
                            str_,
                            this->m_index++
                        });
                        new(_target_out.ptrAt(i)) uint32_t(this->m_index_map[str_]);
                    };   
                }
                return;
            }

            template <typename T>
            auto ColsWithHeader() {
                this->CheckFileEmpty();
                std::unordered_map<std::string_view, lantern::utility::Vector<T>> result_;
                for(auto& first_col : this->m_data.front()) {
                    auto& row = result_[first_col];
                    row = lantern::utility::Vector<T>(this->m_data.size() - 1);
                    row.explicitTotalItem(this->m_data.size() - 1); // reserve space for the column
                    
                    for(auto& str_ : row){
                        std::transform(
                            std::next(this->m_data.begin()),
                            this->m_data.end(),
                            row.begin(),
                            [&](const lantern::string::String& _row) -> T {
                                return _row.as<T>();
                            }
                        );
                    }
                }
                return result_;
            }

            template <typename T>
            auto RowsWithHeader() {
                this->CheckFileEmpty();
                std::unordered_map<std::string, lantern::utility::Vector<T>> result_;
                for(auto& row : this->m_data) {
                    result_[row.front()] = lantern::utility::Vector<T>(row.size() - 1);
                    result_[row.front()].explicitTotalItem(row.size() - 1); // reserve space for the row
                    std::transform(
                        std::next(row.begin()),
                        row.end(),
                        result_[row.front()].begin(),
                        [&](const std::string& _str) -> T {
                            return lantern::utility::ConvertFromString<T>(_str);
                        }
                    );
                }
                return result_;
            }

            template <typename T>
            auto RowsWithHeader(const uint32_t& _index_of_header) {
                this->CheckFileEmpty();
                std::unordered_map<std::string, lantern::utility::Vector<T>> result_;
                for(auto& row : this->m_data) {
                    result_[row[_index_of_header]] = lantern::utility::Vector<T>(row.size() - 1);
                    for(uint32_t i = 0; i < row.size(); i++){
                        if(i == _index_of_header){
                            continue;
                        }
                        result_[row[_index_of_header]].push_back(
                            lantern::utility::ConvertFromString<T>(row[i])
                        );
                    }
                }
                return result_;
            }

            auto GetPtrRow(const uint32_t& _index) {
                if (_index >= this->m_data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }
                return &this->m_data[_index];
            }

            uint32_t GetRowSize() {
                this->CheckFileEmpty();
                return this->m_data.size();
            }

            uint32_t GetColSize(){
                this->CheckFileEmpty();
                return this->m_data.front().size();
            }

        };

        /**
         * @brief Read CSV file to given path
         * @param _path 
         * @return lantern::file::CSVFile
         * @ingroup LanternFile
         */
        [[nodiscard]]
        inline CSVFile ReadCSVFile(const std::filesystem::path& _path) {
           
            CSVFile result_;
            auto& data_ = (*result_.GetDataPtr());
            
            if (!std::filesystem::exists(_path)) {
                throw std::runtime_error(std::format("Error CSVReader, cannot access file path \"{}\" looks like deleted or moved", _path.string()));
            }

            // first check extension
            if (std::filesystem::is_regular_file(_path)) {
                std::string ext_ = _path.extension().string(), col_data_, line_;
                std::transform(ext_.begin(), ext_.end(), ext_.begin(), [](const char& _d) {
                    return std::tolower(_d);
                });
                if (ext_.compare(".csv") == 0) {
                    std::ifstream file_(_path);
                    if (!file_.is_open()) {
                        throw std::runtime_error(std::format("Error CSVReader, failed to open file \"{}\"", ext_));
                    }

                    while (std::getline(file_,line_)) {
                        data_.push_back(lantern::utility::Vector<lantern::string::String>(20));
                        std::stringstream ss(line_);

                        while (std::getline(ss,col_data_,',')) {
                            data_.back().push_back(col_data_);
                        }

                        if (!data_.empty()) {
                            if (data_.front().size() != data_.back().size()) {
                                throw std::runtime_error(std::format("Error ReadCSVFile, the file \"{}\" has different columns sizes", _path.string()));
                            }
                        }
                    }
                    
                }
                else {
                    throw std::runtime_error(std::format("Error CSVReader, the file extension \"{}\" is not json file", ext_));
                }
            }
            else {
                throw std::runtime_error(std::format("Error CSVReader, the path \"{}\" is not file path", _path.string()));
            }

            std::println("File size {}",data_.size());
            return result_;
        }
        /**
         * @brief Read json file from given path
         * @param _path 
         * @return nlohmann::json
         * @ingroup LanternFile
         */
        [[nodiscard]]
        inline nlohmann::json JSONReader(const std::filesystem::path& _path) {

            nlohmann::json result_;

            if (!std::filesystem::exists(_path)) {
                throw std::runtime_error(std::format("Error JSONReader, cannot access file path \"{}\" looks like deleted or moved", _path.string()));
            }

            // first check extension
            if (std::filesystem::is_regular_file(_path)) {
                std::string ext_ = _path.extension().string();
                std::transform(ext_.begin(), ext_.end(), ext_.begin(), [](const char& _d) {
                    return std::tolower(_d);
                });
                if (ext_.compare(".json") == 0) {
                    std::ifstream file_(_path);
                    if (!file_.is_open()) {
                        throw std::runtime_error(std::format("Error JSONReader, failed to open file \"{}\"", ext_));
                    }

                    try {
                        file_ >> result_;
                    }
                    catch (nlohmann::json::parse_error& _err) {
                        throw std::runtime_error(std::format("Error JSONReader, because {}",_err.what()));
                    }
                }
                else {
                    throw std::runtime_error(std::format("Error JSONReader, the file extension \"{}\" is not json file", ext_));
                }
            }
            else {
                throw std::runtime_error(std::format("Error JSONReader, the path \"{}\" is not file path",_path.string()));
            }

            return result_;
        }

        /**
         * @brief HDF5 wrapper for lantern
         */
        class LanternHDF5 {
        private:

            H5std_string m_filename;
            H5::H5File m_file;
            H5::Group m_active_group;
            std::unordered_map<std::string,H5::DataSet> m_datasets;
            std::unordered_map<std::string,H5::DataSpace> m_dataspaces;
            std::unordered_map<std::string,H5::Group> m_groups;
            std::unordered_map<std::string,H5::Attribute> m_attributes;

            void PrintError(const std::string& _message){
                std::println("Error LanternHDF5, {}", _message);
            }

            /**
             * @brief Get all dataset,groups, and attributes from active file
             * @param root_group 
             */
            void GetAllDataInfo(const H5::Group& _root_group) {
                H5::Exception::dontPrint();
                lantern::utility::Vector<H5::Group> groups_stack_ = {_root_group};
                lantern::utility::Vector<std::string> current_path_stack_ = {""};
                std::unordered_set<std::string> visited_groups_;

                while (!groups_stack_.empty()) {
                    H5::Group current_group_ = groups_stack_.back();
                    std::string current_path_ = current_path_stack_.back();
                    groups_stack_.pop_back();
                    current_path_stack_.pop_back();

                    // Skip if already visited
                    if (visited_groups_.contains(current_path_)) continue;
                    visited_groups_.insert(current_path_);

                    // Process datasets and attributes in the current group
                    hsize_t num_objs_ = current_group_.getNumObjs();
                    for (hsize_t i = 0; i < num_objs_; ++i) {
                        std::string obj_name_ = current_group_.getObjnameByIdx(i);
                        H5G_obj_t obj_type_ = current_group_.getObjTypeByIdx(i);

                        switch (obj_type_) {
                            case H5G_GROUP: {
                                std::string new_path_ = current_path_ + "/" + obj_name_;
                                groups_stack_.push_back(current_group_.openGroup(obj_name_));
                                current_path_stack_.push_back(new_path_);
                                this->m_groups.insert({ new_path_, groups_stack_.back()});
                                break;
                            }
                            case H5G_DATASET: {
                                H5::DataSet dataset_ = current_group_.openDataSet(obj_name_);
                                this->m_datasets.insert({ current_path_ + "/" + obj_name_, dataset_});

                                // Process dataset attributes
                                hsize_t num_attrs_ = dataset_.getNumAttrs();
                                for (hsize_t j = 0; j < num_attrs_; ++j) {
                                    H5::Attribute attr_ = dataset_.openAttribute(j);
                                    this->m_attributes.insert({ current_path_ + "/" + obj_name_ + "/"+attr_.getName(), attr_});
                                }
                                break;
                            }
                            default:
                                break;
                        }
                    }
 
                    // Process group attributes
                    hsize_t num_attrs_ = current_group_.getNumAttrs();
                    for (hsize_t j = 0; j < num_attrs_; ++j) {
                        H5::Attribute attr_ = current_group_.openAttribute(j);
                        this->m_attributes.insert({ current_path_ + "/" + attr_.getName(), attr_});
                    }
                }
            }
            
            
        public:



            /**
             * @brief Set active group, the active group is a group which will process all opeartion on the class
             * @param _target_group_name
             */
            void SetActiveGroup(const std::string& _target_group_name){
                try{

                    H5::Exception::dontPrint();

                    if(!this->m_groups.contains(_target_group_name)){
                        throw H5::GroupIException("SetActiveGroup","Selected group ["+_target_group_name+"] does not exists\n");
                    }
                    
                    this->m_active_group = this->m_groups.at(_target_group_name);

                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::GroupIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Get String name of active group 
             * @return std::string
             */
            std::string GetActiveGroupNameAsString() {
                return this->m_active_group.getObjName();
            }

            /**
             * @brief Get All Data such as dataset and attributes
             */
            void GetAllData(){
                try{

                    H5::Exception::dontPrint();
                    
                    if(!this->CheckFileExists()){
                        this->LoadFile(this->m_filename,H5F_ACC_RDWR);
                    }

                    // get root
                    H5::Group root_ = this->m_file.openGroup("/");
                    this->GetAllDataInfo(root_);
                    this->m_active_group = root_;
                    this->m_groups.insert({
                        "/",
                        root_
                    });
                    root_.close();

                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }
            
            LanternHDF5(){}
            LanternHDF5(const H5std_string& _filename) : m_filename(_filename){}

            ~LanternHDF5() {
                for (auto& [_,dataspace] : this->m_dataspaces) {
                    dataspace.close();
                }
                for (auto& [_,dataset] : this->m_datasets) {
                    dataset.close();
                }
                for (auto& [_,group] : this->m_groups) {
                    group.close();
                }
                this->m_file.close();
            }
            
            /**
             * @brief Load file from the filename
             * @param _filename
             * @param AvailableAction
             */
            void LoadFile(std::string _filename,uint32_t AvailableAction){
                try{
                    H5::Exception::dontPrint();
                    this->m_file = H5::H5File(_filename,AvailableAction);
                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Get dataset dimension by dataset name
             * @param _dataset_name
             * @return lantern::utility::Vector<hsize_t>
             */
            lantern::utility::Vector<hsize_t> GetDatasetDims(const std::string& _dataset_name){
                try{
                    H5::Exception::dontPrint();
                    if(!this->m_datasets.contains(_dataset_name)){
                        this->PrintError(std::string("Dataset [")+_dataset_name+"] not found\n");
                        
                    }
                    lantern::utility::Vector<hsize_t> data_;
                    H5::DataSet& dataset_ = this->m_datasets.at(_dataset_name);
                    H5::DataSpace dataspace_ = dataset_.getSpace();
                    uint32_t rank_ = dataspace_.getSimpleExtentNdims();
                    data_ = lantern::utility::Vector<hsize_t>(rank_);
                    data_.explicitTotalItem(rank_);
                    dataspace_.getSimpleExtentDims(data_.data());
                    return data_;
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Get Attribute dimension by attribute name
             * @param _group_name
             * @param _attr_name
             * @return lantern::utility::Vector<hsize_t>
             */
            lantern::utility::Vector<hsize_t> GetAttrDimsAtGroup(const std::string& _group_name,const std::string& _attr_name){
                try{
                    H5::Exception::dontPrint();
                    lantern::utility::Vector<hsize_t> data_;
                    std::string attr_name_ = _group_name + "/" + _attr_name;
                    if(!this->m_attributes.contains(attr_name_)) {
                        this->PrintError(std::string("Attributes [")+_attr_name+"] not found\n");
                        exit(EXIT_FAILURE);
                    }
                    H5::Attribute& attribute_ = this->m_attributes.at(attr_name_);
                    H5::DataSpace dataspace_ = attribute_.getSpace();
                    uint32_t rank_ = dataspace_.getSimpleExtentNdims();
                    data_ = lantern::utility::Vector<hsize_t>(rank_);
                    data_.explicitTotalItem(rank_);
                    dataspace_.getSimpleExtentDims(data_.data());
                    return data_;

                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Get Attribute dimension by attribute name
             * @param _dataset_name
             * @param _attr_name
             * @return lantern::utility::Vector<hsize_t>
             */
            lantern::utility::Vector<hsize_t> GetAttrDimsAtDataset(const std::string& _dataset_name,const std::string& _attr_name) {
                try{
                    H5::Exception::dontPrint();
                    lantern::utility::Vector<hsize_t> data_;
                    std::string attr_name_ = _dataset_name + "/" + _attr_name;
                    if(!this->m_attributes.contains(attr_name_)) {
                        this->PrintError(std::string("Attributes [")+_attr_name+"] not found\n");
                        exit(EXIT_FAILURE);
                    }
                    H5::Attribute& attribute_ = this->m_attributes.at(attr_name_);
                    H5::DataSpace dataspace_ = attribute_.getSpace();
                    uint32_t rank_ = dataspace_.getSimpleExtentNdims();
                    data_ = lantern::utility::Vector<hsize_t>(rank_);
                    data_.explicitTotalItem(rank_);
                    dataspace_.getSimpleExtentDims(data_.data());
                    return data_;
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
                
            }

            /**
             * @brief Print out dataset dimension
             * @param _dataset_name
             */
            void PrintDatasetDims(const std::string& _dataset_name){
                lantern::utility::Vector<hsize_t> dims_ = this->GetDatasetDims(_dataset_name);
                std::cout << std::string(30,'=') << '\n';
                std::cout << "Dataset name : " << _dataset_name << '\n';
                std::cout << "Rank : " << dims_.size() << '\n';
                std::cout << "Dimension : [ ";
                for(auto& p_ : dims_){
                    std::cout << p_ << ' '; 
                }
                std::cout << "]\n";
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Get current file was attach
             * @return H5::H5File&
             */
            H5::H5File& GetFile(){
                return this->m_file;
            }

            /**
             * @brief Print all datasets inside file
             */
            void PrintAllDatasets(){
                std::cout << std::string(30,'=') << '\n';
                std::cout << "All Datasets in file : " << this->m_filename << '\n'; 
                std::cout << std::string(30,'-') << '\n';
                uint32_t i = 0;
                for (auto [name_, dataset_] : this->m_datasets) {
                    std::cout << std::to_string(i) << ". " << name_ << '\n';
                    i++;
                }
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Print all datasets inside file
             */
            void PrintAllAttributes(){
                std::cout << std::string(30,'=') << '\n';
                std::cout << "All Attributes in file : " << this->m_filename << '\n'; 
                std::cout << std::string(30,'-') << '\n';
                uint32_t i = 0;
                for (auto [name_, dataset_] : this->m_attributes) {
                    std::cout << std::to_string(i) << ". " << name_ << '\n';
                    i++;
                }
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Print all datasets inside file
             */
            void PrintAllGroups(){
                std::cout << std::string(30,'=') << '\n';
                std::cout << "All Groups in file : " << this->m_filename << '\n'; 
                std::cout << std::string(30,'-') << '\n';
                uint32_t i = 0;
                for (auto [name_, dataset_] : this->m_groups) {
                    std::cout << std::to_string(i) << ". " << dataset_.getObjName() << '\n';
                    i++;
                }
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Get existed group pointers
             * @param _group_name
             */
            H5::Group* GetGroupPtr(const std::string& _group_name) {
                try {
                    H5::Exception::dontPrint();
                    if (this->m_groups.contains(_group_name)) {
                        throw std::runtime_error(std::format("Cannot get group, the group {} does not exists in file", _group_name));
                    }
                    return &this->m_groups.at(_group_name);
                }
                catch (std::exception& _err) {
                    this->PrintError(_err.what());
                    
                }
            }

            /**
             * @brief Get pointer to groups map
             * @return std::unordered_map<std::string,H5::Group>
             */
            auto* GetGroupsPtr() {
                return &this->m_groups;
            }

            /**
            * @brief Get pointer to groups map
            * @return  std::unordered_map<std::string,H5::Dataset>
            */
            auto* GetDatasetsPtr() {
                return &this->m_datasets;
            }

            /**
             * @brief Create new file, if already exists file with the same name, the file will be replace
             */
            void Create(){
                try{
                    H5::Exception::dontPrint();
                    this->LoadFile(this->m_filename,H5F_ACC_TRUNC);
                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Get all dataspaces
             * @return std::unordered_map<std::string, H5::DataSpace>&
             */
            auto& GetDataSpaces(){
                return this->m_dataspaces;
            }

            /**
             * @brief Create new dataspace
             * @tparam RANK
             * @param _dataspace_name
             * @param _dims
             */
            template <uint32_t RANK>
            void CreateDataSpace(const std::string& _dataspace_name,std::initializer_list<uint64_t> _dims){
                
                uint64_t* dims_ = (uint64_t*)::operator new(sizeof(uint64_t) * _dims.size());
                uint32_t index_ = 0;
                for(auto item_: _dims){
                    new(&dims_[index_++]) uint64_t(std::move(item_));
                }
                
                try{
                    H5::Exception::dontPrint();
                    
                    if(this->CheckFileExists()){
                        if(!this->m_dataspaces.contains(_dataspace_name)){
                            this->m_dataspaces.insert({
                                _dataspace_name,
                                H5::DataSpace(RANK,dims_)
                            });
                            delete dims_;
                        }else{
                            delete dims_;
                            throw H5::DataSpaceIException("DataSpace", "DataSpace ["+_dataspace_name+"] already exists");
                        }
                    }else{
                        delete dims_;
                        throw H5::DataSpaceIException("File", "File does not exists");
                    }
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Create new dataspace
             * @param _dataspace_name
             * @param RANK
             * @param _dims
             */
            void CreateDataSpace(const std::string& _dataspace_name, const uint32_t& RANK,std::initializer_list<uint64_t> _dims){
                
                uint64_t* dims_ = (uint64_t*)::operator new(sizeof(uint64_t) * _dims.size());
                uint32_t index = 0;
                for(auto item: _dims){
                    new(&dims_[index++]) uint64_t(std::move(item));
                }
                
                try{
                    H5::Exception::dontPrint();
                    
                    if(this->CheckFileExists()){
                        if(!this->m_dataspaces.contains(_dataspace_name)){
                            this->m_dataspaces.insert({
                                _dataspace_name,
                                H5::DataSpace(RANK,dims_)
                            });
                            delete dims_;
                        }else{
                            delete dims_;
                            throw H5::DataSpaceIException("DataSpace", "DataSpace ["+_dataspace_name+"] already exists");
                        }
                    }else{
                        delete dims_;
                        throw H5::DataSpaceIException("File", "File does not exists");
                    }
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

             /**
             * @brief Create new dataspace
             * @param _dataspace_name
             */
            void CreateScalarDataSpace(const std::string& _dataspace_name){
                
                try{
                    H5::Exception::dontPrint();
                    
                    if(this->CheckFileExists()){
                        if(!this->m_dataspaces.contains(_dataspace_name)){
                            this->m_dataspaces.insert({
                                _dataspace_name,
                                H5::DataSpace(H5S_SCALAR)
                            });
                        }else{
                            throw H5::DataSpaceIException("DataSpace", "DataSpace ["+_dataspace_name+"] already exists");
                        }
                    }else{
                        throw H5::DataSpaceIException("File", "File does not exists");
                    }
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Create new dataset 
             * @param _dataset_name
             * @param _dataspace_name
             * @param TypeData
             */
            void CreateDataset(const std::string& _dataset_name, const std::string& _dataspace_name, const H5::PredType& _TypeData){
                try{
                    H5::Exception::dontPrint();
                    if(this->CheckFileExists()){

                        std::string dataset_name_ = this->m_active_group.getObjName() + "/" + _dataset_name;

                        if(this->m_datasets.contains(dataset_name_)){
                            throw H5::DataSetIException("Dataset","Dataset ["+dataset_name_+"] already exists");
                        }

                        if(!this->m_dataspaces.contains(_dataspace_name)){
                            throw H5::DataSetIException("Dataset","Dataspace ["+_dataspace_name+"] does not exists");
                        }

                        this->m_datasets.insert({
                            dataset_name_,
                            this->m_active_group.createDataSet(
                                _dataset_name, 
                                _TypeData, 
                                this->m_dataspaces.at(_dataspace_name)
                            )
                        });

                    }else{
                        throw H5::DataSetIException("Dataset","File does not valid");
                    }
                }catch(H5::DataSetIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
                catch (H5::GroupIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Create new dataset
             * @param _dataset_name
             * @param _dataspace_name
             * @param TypeData
             */
            void CreateDataset(const std::string& _dataset_name, const std::string& _dataspace_name, const H5::StrType& _TypeData) {
                try {
                    H5::Exception::dontPrint();
                    if (this->CheckFileExists()) {

                        std::string dataset_name_ = this->m_active_group.getObjName() + "/" + _dataset_name;

                        if (this->m_datasets.contains(dataset_name_)) {
                            throw H5::DataSetIException("Dataset", "Dataset [" + dataset_name_ + "] already exists");
                        }

                        if (!this->m_dataspaces.contains(_dataspace_name)) {
                            throw H5::DataSetIException("Dataset", "Dataspace [" + _dataspace_name + "] does not exists");
                        }

                        this->m_datasets.insert({
                            dataset_name_,
                            this->m_active_group.createDataSet(
                                _dataset_name,
                                _TypeData,
                                this->m_dataspaces.at(_dataspace_name)
                            )
                        });

                    }
                    else {
                        throw H5::DataSetIException("Dataset", "File does not valid");
                    }
                }
                catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
                catch (H5::GroupIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Write dataset, warning this only works if dataset already create or load using GetAllData()
             * @param _dataset_name
             * @param data
             * @param TypeData
             */
            template <typename Data>
            void WriteDataset(const std::string& _dataset_name, Data* _data,  const H5::DataType& _TypeData){
                try{
                    
                    std::string dataset_name_ = this->m_active_group.getObjName() + "/" + _dataset_name;
                    H5::Exception::dontPrint();
                    if(!this->m_datasets.contains(dataset_name_)){
                        throw H5::DataSetIException("WriteDataset","Cannot find dataset ["+dataset_name_+"]\n");
                    }

                    H5::DataSet dataset_ = this->m_datasets.at(dataset_name_);
                    dataset_.write(_data, _TypeData);                        

                }catch(H5::DataSetIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Write dataset, warning this only works if dataset already create or load using GetAllData()
             * @param _dataset_name
             * @param data
             * @param TypeData
             */
            void WriteDataset(const std::string& _dataset_name,const std::string& _data, const H5::StrType& _TypeData) {
                try {

                    std::string dataset_name_ = this->m_active_group.getObjName() + "/" + _dataset_name;
                    H5::Exception::dontPrint();
                    if (!this->m_datasets.contains(dataset_name_)) {
                        throw H5::DataSetIException("WriteDataset", "Cannot find dataset [" + dataset_name_ + "]\n");
                    }

                    H5::DataSet dataset_ = this->m_datasets.at(dataset_name_);
                    dataset_.write(_data, _TypeData);

                }
                catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
                catch (H5::FileIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief check if dataset exists, if you want to load and check the datasets exists don't forget to call GetAllData() first
             * @param _dataset_name
             * @return bool
             */
            bool CheckDataSetExists(const std::string& _dataset_name){

                return this->m_datasets.contains(_dataset_name);

            }

            /**
             * @brief check if attribute exists, if you want to load and check the attr exists don't forget to call GetAllData() first
             * @param _attr_name
             * @return bool
             */
            bool CheckAttributeExists(const std::string& _attr_name) {

                return this->m_attributes.contains(_attr_name);

            }

            /**
             * @brief Check if current file was load into class
             * @return bool
             */
            bool CheckFileExists(){
                return this->m_file.isValid(this->m_file.getId());
            }

            /**
             * @brief Create new attribute at dataset
             * @tparam DataType
             * @param _dataset_name
             * @param _dataspace_name
             * @param _attr_name
             * @param _datatype
             */
            template <typename DataType>
            void CreateAttributeAtDataset(
                const std::string& _dataset_name, 
                const std::string& _dataspace_name, 
                const std::string& _attr_name,
                const DataType& _datatype
            ){
                try{
                    
                    H5::Exception::dontPrint();
                    std::string dataset_name_ = this->m_active_group.getObjName() + "/" + _dataset_name;
                    std::string attr_name_ = dataset_name_ + "/" + _attr_name;
                    
                    if(this->m_attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] already exists\n");
                    }
                    
                    if(!this->m_datasets.contains(dataset_name_)){
                        throw H5::AttributeIException("CreateAttribute","Cannot find dataset ["+dataset_name_+"]\n");
                    }

                    if(!this->m_dataspaces.contains(_dataspace_name)){
                        throw H5::AttributeIException("CreateAttribute","Cannot find dataspace ["+_dataspace_name+"]\n");
                    }

                    H5::DataSet dataset_ = this->m_datasets.at(dataset_name_);
                    H5::DataSpace dataspace_ = this->m_dataspaces.at(_dataspace_name);
                    
                    this->m_attributes.insert({
                        attr_name_,
                        dataset_.createAttribute(_attr_name,_datatype,dataspace_)
                    });

                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Create new attribute at group
             * @tparam DataType
             * @param _group_name
             * @param _dataspace_name
             * @param _attr_name
             * @param _datatype
             */
            template <typename DataType>
            void CreateAttributeAtGroup(
                const std::string& _group_name, 
                const std::string& _dataspace_name, 
                const std::string& _attr_name,
                const DataType& _datatype
            ){
                try{
                    
                    H5::Exception::dontPrint();
                    std::string attr_name_ = _group_name + "/" + _attr_name;
                    
                    if(this->m_attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] already exists\n");
                    }

                    if(!this->m_groups.contains(_group_name)){
                        throw H5::AttributeIException("CreateAttribute","Group ["+_group_name+"] does not exists\n");
                    }

                    if(!this->m_dataspaces.contains(_dataspace_name)){
                        throw H5::AttributeIException("CreateAttribute","Cannot find dataspace ["+_dataspace_name+"]\n");
                    }

                    H5::Group group_ = this->m_groups.at(_group_name);
                    H5::DataSpace dataspace_ = this->m_dataspaces.at(_dataspace_name);
                    
                    this->m_attributes.insert({
                        attr_name_,
                        group_.createAttribute(_attr_name,_datatype,dataspace_)
                    });

                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::DataSpaceIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Write existsing attribute, warning this only works if attributes already create or load using GetAllData()
             * @tparam DataType
             * @tparam Data
             * @param _dataset_name
             * @param _attr_name
             * @param _datatype
             * @param _data
             */
            template <typename DataType, typename Data>
            void WriteAttributeAtDataset(
                const std::string& _dataset_name,
                const std::string& _attr_name,
                const DataType& _datatype,
                Data _data
            ){
                try{
                    
                    H5::Exception::dontPrint();
                    std::string dataset_name_ = this->m_active_group.getObjName() + "/" + _dataset_name;
                    std::string attr_name_ = dataset_name_ + "/" + _attr_name;

                    if(!this->m_attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }

                    H5::Attribute attr_ = this->m_attributes.at(attr_name_);
                    attr_.write(_datatype, _data);

                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Write existsing attribute, warning this only works if attributes already create or load using GetAllData()
             * @tparam DataType
             * @tparam Data
             * @param _group_name
             * @param _attr_name
             * @param _datatype
             * @param _data
             */
            template <typename DataType, typename Data>
            void WriteAttributeAtGroup(
                const std::string& _group_name,
                const std::string& _attr_name,
                const DataType& _datatype,
                Data _data
            ){
                try{
                    
                    H5::Exception::dontPrint();
                    std::string attr_name_ = _group_name + "/" + _attr_name;

                    if(!this->m_attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }

                    H5::Attribute attr_ = this->m_attributes.at(attr_name_);
                    attr_.write(_datatype, _data);

                }catch(H5::AttributeIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Read attribute from group, warning this only works if attribute already create or load using GetAllData()
             * @tparam DataType
             * @param _dataset_name
             * @param _attr_name
             * @param _datatype
             * @param _data
             */
            template <typename DataType>
            void ReadAttributeAtDataset(
                const std::string& _dataset_name,
                const std::string& _attr_name,
                const DataType& _datatype,
                std::string& _data
            ){
                try{
                    
                    H5::Exception::dontPrint();
                    std::string attr_name_ = _dataset_name + "/" + _attr_name;

                    if(!this->m_attributes.contains(attr_name_)){
                        throw H5::DataSetIException("ReadAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }
                    
                    H5::Attribute attr_ = this->m_attributes.at(attr_name_);
                    attr_.read(_datatype, _data);

                }catch(H5::DataSetIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Read attribute from group, warning this only works if attribute already create or load using GetAllData()
             * @tparam DataType
             * @tparam Data
             * @param _dataset_name
             * @param _attr_name
             * @param _datatype
             * @param _data
             */
            template <typename DataType, typename Data>
            void ReadAttributeAtDataset(
                const std::string& _dataset_name,
                const std::string& _attr_name,
                const DataType& _datatype,
                Data* _data
            ) {
                try {

                    H5::Exception::dontPrint();
                    std::string attr_name_ = _dataset_name + "/" + _attr_name;

                    if (!this->m_attributes.contains(attr_name_)) {
                        throw H5::DataSetIException("ReadAttribute", "Attribute [" + attr_name_ + "] does not exists\n");
                    }

                    H5::Attribute attr_ = this->m_attributes.at(attr_name_);
                    attr_.read(_datatype, _data);

                }
                catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Read attribute from group, warning this only works if attribute already create or load using GetAllData()
             * @param _group_name
             * @param _attr_name
             * @param _datatype
             * @param _data
             */
            void ReadAttributeAtGroup(
                const std::string& _group_name,
                const std::string& _attr_name,
                const H5::StrType& _datatype,
                std::string& _data
            ) {
                try {

                    H5::Exception::dontPrint();
                    std::string attr_name_ = _group_name + "/" + _attr_name;

                    if (!this->m_attributes.contains(attr_name_)) {
                        throw H5::DataSetIException("ReadAttribute", "Attribute [" + attr_name_ + "] does not exists\n");
                    }

                    H5::Attribute attr_ = this->m_attributes.at(attr_name_);
                    attr_.read(_datatype,_data);

                }
                catch (H5::AttributeIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                   
                }
            }

            /**
            * @brief Read attribute from group, warning this only works if attribute already create or load using GetAllData()
            * @tparam DataType
            * @tparam Data
            * @param _group_name
            * @param _attr_name
            * @param _datatype
            * @param _data
            */
            template <typename DataType, typename Data>
            void ReadAttributeAtGroup(
                const std::string& _group_name,
                const std::string& _attr_name,
                const DataType& _datatype,
                Data* _data
            ) {
                try {

                    H5::Exception::dontPrint();
                    std::string attr_name_ = _group_name + "/" + _attr_name;

                    if (!this->m_attributes.contains(attr_name_)) {
                        throw H5::DataSetIException("ReadAttribute", "Attribute [" + attr_name_ + "] does not exists\n");
                    }

                    H5::Attribute attr_ = this->m_attributes.at(attr_name_);
                    attr_.read(_datatype, _data);

                }
                catch (H5::AttributeIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }
            
            /**
             * @brief Read exisiting attribute, warning this only works if attribute already create or load using GetAllData()
             * @tparam Data
             * @param _dataset_name
             * @param data
             * @param TypeData
             */
            template <typename Data>
            void ReadDataset(const std::string& _dataset_name, Data* data, const H5::PredType& TypeData) {
                try {

                    H5::Exception::dontPrint();
                    if (!this->m_datasets.contains(_dataset_name)) {
                        throw H5::DataSetIException("ReadDataset", "Cannot find dataset [" + _dataset_name + "]\n");
                    }

                    H5::DataSet dataset_ = this->m_datasets.at(_dataset_name);
                    dataset_.read(data, TypeData);

                }
                catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
                catch (H5::FileIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

             /**
             * @brief Read exisiting attribute, warning this only works if attribute already create or load using GetAllData()
             * @param _dataset_name
             * @param data
             * @param TypeData
             */
            void ReadDataset(const std::string & _dataset_name, std::string& data, const H5::StrType& TypeData) {
                try {

                    H5::Exception::dontPrint();
                    if (!this->m_datasets.contains(_dataset_name)) {
                        throw H5::DataSetIException("ReadDataset", "Cannot find dataset [" + _dataset_name + "]\n");
                    }

                    H5::DataSet dataset_ = this->m_datasets.at(_dataset_name);
                    dataset_.read(data, TypeData);

                }
                catch (H5::DataSetIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
                catch (H5::FileIException& _err) {
                    this->PrintError(_err.getDetailMsg());
                    
                }
            }

            /**
             * @brief Check if group exists
             * @param _group_name 
             * @return bool
             */
            bool CheckGroupExists(const std::string& _group_name) {
               
                if (!this->m_groups.contains(_group_name)) {
                    return false;
                }

                return true;

            }

            /**
             * @brief Create new group ad current active group
             * @param _group_name 
             */
            void CreateGroup(const std::string& _group_name){

                try{

                    H5::Exception::dontPrint();

                    if(this->m_groups.contains(_group_name)){
                        throw H5::GroupIException("CreateGroup","Group ["+_group_name+"] already exists\n");
                    }

                    H5::Group group_ = this->m_active_group.createGroup(_group_name);
                    this->m_groups.insert({
                        _group_name,
                        group_
                    });

                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::GroupIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }

            }

            /**
             * @brief Create group explicitly at given path
             * @param _target_group_name 
             * @param _group_name 
             */
            void CreateGroupAt(const std::string& _target_group_name,const std::string& _group_name){

                try{

                    H5::Exception::dontPrint();

                    if(!this->m_groups.contains(_target_group_name)){
                        throw H5::GroupIException("CreateGroup","Target group ["+_group_name+"] does not exists\n");
                    }
                    if(this->m_groups.contains(_group_name)){
                        throw H5::GroupIException("CreateGroup","Group ["+_group_name+"] already exists\n");
                    }

                    H5::Group group_ = this->m_groups.at(_target_group_name);
                    H5::Group new_group_ = group_.createGroup(_group_name);
                    this->m_groups.insert({
                        _group_name,
                        group_
                    });

                }catch(H5::FileIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }catch(H5::GroupIException& _err){
                    this->PrintError(_err.getDetailMsg());
                    
                }

            }



        };

    }

}