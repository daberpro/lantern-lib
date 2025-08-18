#pragma once
#include "../pch.h"
#include "Vector.h"
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
            lantern::utility::Vector<lantern::utility::Vector<std::string>> data;

            /**
             * @brief Convert string into T type
             * @tparam T 
             * @param _str 
             * @return T
             */
            template <typename T>
            T ConvertFromString(const std::string& _str) {
                if constexpr (std::is_arithmetic_v<T>) {
                    T value{};
                    auto [ptr, e] = std::from_chars(_str.data(),_str.data() + _str.size(),value);
                    if (e != std::errc{}) {
                        std::println("{}",_str);
                        throw std::runtime_error("Error CSVFile, ivalid conversion");
                    }
                    return value;
                }
                else {
                    if constexpr (std::is_same_v<T,std::string>) {
                        return _str;
                    }
                    else {
                        T value{};
                        std::istringstream iss(_str);
                        iss >> value;
                        if (iss.fail() || !iss.eof()) {
                            throw std::runtime_error("Error CSVFile,conversion failed or extra characters found.");
                        }
                        return value;
                    }
                }
            }

            void CheckFileEmpty(){
                if(this->data.empty()){
                    throw std::runtime_error(std::format("Error CSVFile, file is empty"));
                }
            }

        public:

            CSVFile(){}
            CSVFile(CSVFile&& _file) noexcept {
                this->data.movePtrData(_file.data);
            }

            void operator =(CSVFile&& _file) noexcept {
                this->data.movePtrData(_file.data);
            }
            /**
             * @brief Get pointer to data inside CSV file
             * @return lantern::utility::Vector<lantern::utility::Vector<std::string>>
             */
            auto* GetDataPtr() {
                return &this->data;
            }

            /**
             * @brief Get data in specific row, col and cast to type T
             * @tparam T 
             * @param row 
             * @param col 
             * @return T
             */
            template <typename T>
            T Get(const uint32_t& row, const uint32_t col) {
                this->CheckFileEmpty();
                if (col >= this->data.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access column index \"{}\" out of bound", col));
                }
                if (row >= this->data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", row));
                }
                return this->ConvertFromString<T>(this->data[row][col]);
            }

            /**
             * @brief Get column at index and cast to T type
             * @tparam T 
             * @param _index 
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Col(const uint32_t& _index) {
                this->CheckFileEmpty();
                if (_index >= this->data.front().size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access column index \"{}\" out of bound", _index));
                }
                lantern::utility::Vector<T> result(this->data.size());
                result.explicitTotalItem(this->data.size());
                std::transform(
                    this->data.begin(),
                    this->data.end(),
                    result.begin(),
                    [&](const lantern::utility::Vector<std::string>& row) -> T {
                        return this->ConvertFromString<T>(row[_index]);
                    }
                );
                return result;
            }

            /**
             * @brief Get row at index and cast to T type
             * @tparam T
             * @param _index
             * @return lantern::utility::Vector<T>
             */
            template <typename T>
            auto Row(const uint32_t& _index) {
                this->CheckFileEmpty();
                if (_index >= this->data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }

                auto& data_ = this->data[_index];
                uint32_t allocated_size = data_.size();
                lantern::utility::Vector<T> result(allocated_size);
                result.explicitTotalItem(allocated_size);
                std::transform(
                    data_.begin(),
                    data_.end(),
                    result.begin(),
                    [&](const std::string_view& _str) -> T {
                        return this->ConvertFromString<T>(_str);
                    }
                );
                return result;
            }

            template <typename T>
            auto ColsWithHeader() {
                this->CheckFileEmpty();
                std::unordered_map<std::string, lantern::utility::Vector<T>> result;
                for(auto& col : this->data.front()) {
                    result[col] = lantern::utility::Vector<T>(this->data.size() - 1);
                    result[col].explicitTotalItem(this->data.size() - 1); // reserve space for the column
                    std::transform(
                        std::next(this->data.begin()),
                        this->data.end(),
                        result[col].begin(),
                        [&](const lantern::utility::Vector<std::string>& row) -> T {
                            return this->ConvertFromString<T>(row[col]);
                        }
                    );
                }
                return result;
            }

            template <typename T>
            auto RowsWithHeader() {
                this->CheckFileEmpty();
                std::unordered_map<std::string, lantern::utility::Vector<T>> result;
                for(auto& row : this->data) {
                    result[row.front()] = lantern::utility::Vector<T>(row.size() - 1);
                    result[row.front()].explicitTotalItem(row.size() - 1); // reserve space for the row
                    std::transform(
                        std::next(row.begin()),
                        row.end(),
                        result[row.front()].begin(),
                        [&](const std::string& _str) -> T {
                            return this->ConvertFromString<T>(_str);
                        }
                    );
                }
                return result;
            }

            auto GetPtrRow(const uint32_t& _index) {
                if (_index >= this->data.size()) {
                    throw std::runtime_error(std::format("Error CSVFile, cannot access row index \"{}\" out of bound", _index));
                }
                return &this->data[_index];
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
           
            CSVFile result;
            auto& data = (*result.GetDataPtr());
            
            if (!std::filesystem::exists(_path)) {
                throw std::runtime_error(std::format("Error CSVReader, cannot access file path \"{}\" looks like deleted or moved", _path.string()));
            }

            // first check extension
            if (std::filesystem::is_regular_file(_path)) {
                std::string ext = _path.extension().string(), col_data, line;
                std::transform(ext.begin(), ext.end(), ext.begin(), [](const char& d) {
                    return std::tolower(d);
                });
                if (ext.compare(".csv") == 0) {
                    std::ifstream file(_path);
                    if (!file.is_open()) {
                        throw std::runtime_error(std::format("Error CSVReader, failed to open file \"{}\"", ext));
                    }

                    while (std::getline(file,line)) {
                        data.push_back(lantern::utility::Vector<std::string>(20));
                        std::stringstream ss(line);

                        while (std::getline(ss,col_data,',')) {
                            data.back().push_back(col_data);
                        }

                        if (!data.empty()) {
                            if (data.front().size() != data.back().size()) {
                                throw std::runtime_error(std::format("Error ReadCSVFile, the file \"{}\" has different columns sizes", _path.string()));
                            }
                        }
                    }
                    
                }
                else {
                    throw std::runtime_error(std::format("Error CSVReader, the file extension \"{}\" is not json file", ext));
                }
            }
            else {
                throw std::runtime_error(std::format("Error CSVReader, the path \"{}\" is not file path", _path.string()));
            }

            return result;
        }
        /**
         * @brief Read json file from given path
         * @param _path 
         * @return nlohmann::json
         * @ingroup LanternFile
         */
        [[nodiscard]]
        inline nlohmann::json JSONReader(const std::filesystem::path& _path) {

            nlohmann::json result;

            if (!std::filesystem::exists(_path)) {
                throw std::runtime_error(std::format("Error JSONReader, cannot access file path \"{}\" looks like deleted or moved", _path.string()));
            }

            // first check extension
            if (std::filesystem::is_regular_file(_path)) {
                std::string ext = _path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](const char& d) {
                    return std::tolower(d);
                });
                if (ext.compare(".json") == 0) {
                    std::ifstream file(_path);
                    if (!file.is_open()) {
                        throw std::runtime_error(std::format("Error JSONReader, failed to open file \"{}\"", ext));
                    }

                    try {
                        file >> result;
                    }
                    catch (nlohmann::json::parse_error& err) {
                        throw std::runtime_error(std::format("Error JSONReader, because {}",err.what()));
                    }
                }
                else {
                    throw std::runtime_error(std::format("Error JSONReader, the file extension \"{}\" is not json file", ext));
                }
            }
            else {
                throw std::runtime_error(std::format("Error JSONReader, the path \"{}\" is not file path",_path.string()));
            }

            return result;
        }

        /**
         * @brief HDF5 wrapper for lantern
         */
        class LanternHDF5 {
        private:

            H5std_string filename;
            H5::H5File file;
            H5::Group active_group;
            std::unordered_map<std::string,H5::DataSet> datasets;
            std::unordered_map<std::string,H5::DataSpace> dataspaces;
            std::unordered_map<std::string,H5::Group> groups;
            std::unordered_map<std::string,H5::Attribute> attributes;

            /**
             * @brief Get all dataset,groups, and attributes from active file
             * @param root_group 
             */
            void GetAllDataInfo(const H5::Group& root_group) {
                lantern::utility::Vector<H5::Group> groups_stack = {root_group};
                lantern::utility::Vector<std::string> current_path_stack = {""};
                std::unordered_set<std::string> visited_groups;

                while (!groups_stack.empty()) {
                    H5::Group current_group = groups_stack.back();
                    std::string current_path = current_path_stack.back();
                    groups_stack.pop_back();
                    current_path_stack.pop_back();

                    // Skip if already visited
                    if (visited_groups.contains(current_path)) continue;
                    visited_groups.insert(current_path);

                    // Process datasets and attributes in the current group
                    hsize_t num_objs = current_group.getNumObjs();
                    for (hsize_t i = 0; i < num_objs; ++i) {
                        std::string obj_name = current_group.getObjnameByIdx(i);
                        H5G_obj_t obj_type = current_group.getObjTypeByIdx(i);

                        switch (obj_type) {
                            case H5G_GROUP: {
                                std::string new_path = current_path + "/" + obj_name;
                                groups_stack.push_back(current_group.openGroup(obj_name));
                                current_path_stack.push_back(new_path);
                                this->groups.insert({ new_path, groups_stack.back()});
                                break;
                            }
                            case H5G_DATASET: {
                                H5::DataSet dataset = current_group.openDataSet(obj_name);
                                this->datasets.insert({ current_path + "/" + obj_name, dataset});

                                // Process dataset attributes
                                hsize_t num_attrs = dataset.getNumAttrs();
                                for (hsize_t j = 0; j < num_attrs; ++j) {
                                    H5::Attribute attr = dataset.openAttribute(j);
                                    this->attributes.insert({ current_path + "/" + obj_name + "/"+attr.getName(), attr});
                                }
                                break;
                            }
                            default:
                                break;
                        }
                    }
 
                    // Process group attributes
                    hsize_t num_attrs = current_group.getNumAttrs();
                    for (hsize_t j = 0; j < num_attrs; ++j) {
                        H5::Attribute attr = current_group.openAttribute(j);
                        this->attributes.insert({ current_path + "/" + attr.getName(), attr});
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

                    if(!this->groups.contains(_target_group_name)){
                        throw H5::GroupIException("SetActiveGroup","Selected group ["+_target_group_name+"] does not exists\n");
                    }
                    
                    this->active_group = this->groups.at(_target_group_name);

                }catch(H5::FileIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::GroupIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Get String name of active group 
             * @return std::string
             */
            std::string GetActiveGroupNameAsString() {
                return this->active_group.getObjName();
            }

            /**
             * @brief Get All Data such as dataset and attributes
             */
            void GetAllData(){
                try{
                    
                    if(!this->CheckFileExists()){
                        this->LoadFile(this->filename,H5F_ACC_RDWR);
                    }

                    // get root
                    H5::Group root = this->file.openGroup("/");
                    this->GetAllDataInfo(root);
                    this->active_group = root;
                    this->groups.insert({
                        "/",
                        root
                    });
                    root.close();

                }catch(H5::FileIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }
            
            LanternHDF5(){}
            LanternHDF5(const H5std_string& _filename) : filename(_filename){}

            ~LanternHDF5() {
                for (auto& [_,dataspace] : this->dataspaces) {
                    dataspace.close();
                }
                for (auto& [_,dataset] : this->datasets) {
                    dataset.close();
                }
                for (auto& [_,group] : this->groups) {
                    group.close();
                }
                this->file.close();
            }
            
            /**
             * @brief Load file from the filename
             * @param _filename
             * @param AvailableAction
             */
            void LoadFile(std::string _filename,uint32_t AvailableAction){
                try{
                    this->file = H5::H5File(_filename,AvailableAction);
                }catch(H5::FileIException& err){
                    std::println("Error File, Cannot open [{}] maybe deleted or modified", _filename);
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Get dataset dimension by dataset name
             * @param _dataset_name
             * @return lantern::utility::Vector<hsize_t>
             */
            lantern::utility::Vector<hsize_t> GetDatasetDims(const std::string& _dataset_name){
                lantern::utility::Vector<hsize_t> data;
                if(!this->datasets.contains(_dataset_name)){
                    std::cout << "Dataset ["+_dataset_name+"] not found\n";
                    data.clean();
                    return data;   
                }
                H5::DataSet& dataset = this->datasets.at(_dataset_name);
                H5::DataSpace dataspace = dataset.getSpace();
                uint32_t rank = dataspace.getSimpleExtentNdims();
                data = lantern::utility::Vector<hsize_t>(rank);
                dataspace.getSimpleExtentDims(data.getData());
                data.explicitTotalItem(rank);
                return data;
            }

            /**
             * @brief Get Attribute dimension by attribute name
             * @param _group_name
             * @param _attr_name
             * @return lantern::utility::Vector<hsize_t>
             */
            lantern::utility::Vector<hsize_t> GetAttrDimsAtGroup(const std::string& _group_name,const std::string& _attr_name){
                lantern::utility::Vector<hsize_t> data;
                std::string attr_name_ = _group_name + "/" + _attr_name;
                if(!this->attributes.contains(attr_name_)) {
                    std::cout << "Attr [" + attr_name_ + "] not found\n";
                    data.clean();
                    return data;
                }
                H5::Attribute& attribute = this->attributes.at(attr_name_);
                H5::DataSpace dataspace = attribute.getSpace();
                uint32_t rank = dataspace.getSimpleExtentNdims();
                data = lantern::utility::Vector<hsize_t>(rank);
                dataspace.getSimpleExtentDims(data.getData());
                data.explicitTotalItem(rank);
                return data;
            }

            /**
             * @brief Get Attribute dimension by attribute name
             * @param _dataset_name
             * @param _attr_name
             * @return lantern::utility::Vector<hsize_t>
             */
            lantern::utility::Vector<hsize_t> GetAttrDimsAtDataset(const std::string& _dataset_name,const std::string& _attr_name) {
                lantern::utility::Vector<hsize_t> data;
                std::string attr_name_ = _dataset_name + "/" + _attr_name;
                if (!this->attributes.contains(attr_name_)) {
                    std::cout << "Attr [" + attr_name_ + "] not found\n";
                    data.clean();
                    return data;
                }
                H5::Attribute& attribute = this->attributes.at(attr_name_);
                H5::DataSpace dataspace = attribute.getSpace();
                uint32_t rank = dataspace.getSimpleExtentNdims();
                data = lantern::utility::Vector<hsize_t>(rank);
                dataspace.getSimpleExtentDims(data.getData());
                data.explicitTotalItem(rank);
                return data;
            }

            /**
             * @brief Print out dataset dimension
             * @param _dataset_name
             */
            void PrintDatasetDims(const std::string& _dataset_name){
                lantern::utility::Vector<hsize_t> dims = this->GetDatasetDims(_dataset_name);
                std::cout << std::string(30,'=') << '\n';
                std::cout << "Dataset name : " << _dataset_name << '\n';
                std::cout << "Rank : " << dims.size() << '\n';
                std::cout << "Dimension : [ ";
                for(auto& p : dims){
                    std::cout << p << ' '; 
                }
                std::cout << "]\n";
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Get current file was attach
             * @return H5::H5File&
             */
            H5::H5File& GetFile(){
                return this->file;
            }

            /**
             * @brief Print all datasets inside file
             */
            void PrintAllDatasets(){
                std::cout << std::string(30,'=') << '\n';
                std::cout << "All Datasets in file : " << this->filename << '\n'; 
                std::cout << std::string(30,'-') << '\n';
                uint32_t i = 0;
                for (auto [name, dataset] : this->datasets) {
                    std::cout << std::to_string(i) << ". " << name << '\n';
                    i++;
                }
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Print all datasets inside file
             */
            void PrintAllAttributes(){
                std::cout << std::string(30,'=') << '\n';
                std::cout << "All Attributes in file : " << this->filename << '\n'; 
                std::cout << std::string(30,'-') << '\n';
                uint32_t i = 0;
                for (auto [name, dataset] : this->attributes) {
                    std::cout << std::to_string(i) << ". " << name << '\n';
                    i++;
                }
                std::cout << std::string(30,'=') << '\n';
            }

            /**
             * @brief Print all datasets inside file
             */
            void PrintAllGroups(){
                std::cout << std::string(30,'=') << '\n';
                std::cout << "All Groups in file : " << this->filename << '\n'; 
                std::cout << std::string(30,'-') << '\n';
                uint32_t i = 0;
                for (auto [name, dataset] : this->groups) {
                    std::cout << std::to_string(i) << ". " << dataset.getObjName() << '\n';
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
                    if (this->groups.contains(_group_name)) {
                        throw std::runtime_error(std::format("Cannot get group, the group {} does not exists in file", _group_name));
                    }
                    return &this->groups.at(_group_name);
                }
                catch (std::exception& err) {
                    std::println("Lantern Error, {}", err.what());
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Get pointer to groups map
             * @return std::unordered_map<std::string,H5::Group>
             */
            auto* GetGroupsPtr() {
                return &this->groups;
            }

            /**
            * @brief Get pointer to groups map
            * @return  std::unordered_map<std::string,H5::Dataset>
            */
            auto* GetDatasetsPtr() {
                return &this->datasets;
            }

            /**
             * @brief Create new file, if already exists file with the same name, the file will be replace
             */
            void Create(){
                try{
                    H5::Exception::dontPrint();
                    this->LoadFile(this->filename,H5F_ACC_TRUNC);
                }catch(H5::FileIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Get all dataspaces
             * @return std::unordered_map<std::string, H5::DataSpace>&
             */
            auto& GetDataSpaces(){
                return this->dataspaces;
            }

            /**
             * @brief Create new dataspace
             * @tparam RANK
             * @param _dataspace_name
             * @param _dims
             */
            template <uint32_t RANK>
            void CreateDataSpace(const std::string& _dataspace_name,std::initializer_list<uint64_t> _dims){
                
                uint64_t* dims = (uint64_t*)::operator new(sizeof(uint64_t) * _dims.size());
                uint32_t index = 0;
                for(auto item: _dims){
                    new(&dims[index++]) uint64_t(std::move(item));
                }
                
                try{
                    H5::Exception::dontPrint();
                    
                    if(this->CheckFileExists()){
                        if(!this->dataspaces.contains(_dataspace_name)){
                            this->dataspaces.insert({
                                _dataspace_name,
                                H5::DataSpace(RANK,dims)
                            });
                            delete dims;
                        }else{
                            delete dims;
                            throw H5::DataSpaceIException("DataSpace", "DataSpace ["+_dataspace_name+"] already exists");
                        }
                    }else{
                        delete dims;
                        throw H5::DataSpaceIException("File", "File does not exists");
                    }
                }catch(H5::DataSpaceIException& err){
                    std::cout << err.getCDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Create new dataspace
             * @param _dataspace_name
             * @param RANK
             * @param _dims
             */
            void CreateDataSpace(const std::string& _dataspace_name, const uint32_t& RANK,std::initializer_list<uint64_t> _dims){
                
                uint64_t* dims = (uint64_t*)::operator new(sizeof(uint64_t) * _dims.size());
                uint32_t index = 0;
                for(auto item: _dims){
                    new(&dims[index++]) uint64_t(std::move(item));
                }
                
                try{
                    H5::Exception::dontPrint();
                    
                    if(this->CheckFileExists()){
                        if(!this->dataspaces.contains(_dataspace_name)){
                            this->dataspaces.insert({
                                _dataspace_name,
                                H5::DataSpace(RANK,dims)
                            });
                            delete dims;
                        }else{
                            delete dims;
                            throw H5::DataSpaceIException("DataSpace", "DataSpace ["+_dataspace_name+"] already exists");
                        }
                    }else{
                        delete dims;
                        throw H5::DataSpaceIException("File", "File does not exists");
                    }
                }catch(H5::DataSpaceIException& err){
                    std::cout << err.getCDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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
                        if(!this->dataspaces.contains(_dataspace_name)){
                            this->dataspaces.insert({
                                _dataspace_name,
                                H5::DataSpace(H5S_SCALAR)
                            });
                        }else{
                            throw H5::DataSpaceIException("DataSpace", "DataSpace ["+_dataspace_name+"] already exists");
                        }
                    }else{
                        throw H5::DataSpaceIException("File", "File does not exists");
                    }
                }catch(H5::DataSpaceIException& err){
                    std::cout << err.getCDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Create new dataset 
             * @param _dataset_name
             * @param _dataspace_name
             * @param TypeData
             */
            void CreateDataset(const std::string& _dataset_name, const std::string& _dataspace_name, const H5::PredType& TypeData){
                try{
                    H5::Exception::dontPrint();
                    if(this->CheckFileExists()){

                        std::string dataset_name_ = this->active_group.getObjName() + "/" + _dataset_name;

                        if(this->datasets.contains(dataset_name_)){
                            throw H5::DataSetIException("Dataset","Dataset ["+dataset_name_+"] already exists");
                        }

                        if(!this->dataspaces.contains(_dataspace_name)){
                            throw H5::DataSetIException("Dataset","Dataspace ["+_dataspace_name+"] does not exists");
                        }

                        this->datasets.insert({
                            dataset_name_,
                            this->active_group.createDataSet(
                                _dataset_name, 
                                TypeData, 
                                this->dataspaces.at(_dataspace_name)
                            )
                        });

                    }else{
                        throw H5::DataSetIException("Dataset","File does not valid");
                    }
                }catch(H5::DataSetIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
                catch (H5::GroupIException& err) {
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Create new dataset
             * @param _dataset_name
             * @param _dataspace_name
             * @param TypeData
             */
            void CreateDataset(const std::string& _dataset_name, const std::string& _dataspace_name, const H5::StrType& TypeData) {
                try {
                    H5::Exception::dontPrint();
                    if (this->CheckFileExists()) {

                        std::string dataset_name_ = this->active_group.getObjName() + "/" + _dataset_name;

                        if (this->datasets.contains(dataset_name_)) {
                            throw H5::DataSetIException("Dataset", "Dataset [" + dataset_name_ + "] already exists");
                        }

                        if (!this->dataspaces.contains(_dataspace_name)) {
                            throw H5::DataSetIException("Dataset", "Dataspace [" + _dataspace_name + "] does not exists");
                        }

                        this->datasets.insert({
                            dataset_name_,
                            this->active_group.createDataSet(
                                _dataset_name,
                                TypeData,
                                this->dataspaces.at(_dataspace_name)
                            )
                            });

                    }
                    else {
                        throw H5::DataSetIException("Dataset", "File does not valid");
                    }
                }
                catch (H5::DataSetIException& err) {
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
                catch (H5::GroupIException& err) {
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Write dataset, warning this only works if dataset already create or load using GetAllData()
             * @param _dataset_name
             * @param data
             * @param TypeData
             */
            template <typename Data>
            void WriteDataset(const std::string& _dataset_name, Data* data,  const H5::DataType& TypeData){
                try{
                    
                    std::string dataset_name_ = this->active_group.getObjName() + "/" + _dataset_name;
                    H5::Exception::dontPrint();
                    if(!this->datasets.contains(dataset_name_)){
                        throw H5::DataSetIException("WriteDataset","Cannot find dataset ["+dataset_name_+"]\n");
                    }

                    H5::DataSet dataset_ = this->datasets.at(dataset_name_);
                    dataset_.write(data, TypeData);                        

                }catch(H5::DataSetIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::FileIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Write dataset, warning this only works if dataset already create or load using GetAllData()
             * @param _dataset_name
             * @param data
             * @param TypeData
             */
            void WriteDataset(const std::string& _dataset_name,const std::string& data, const H5::StrType& TypeData) {
                try {

                    std::string dataset_name_ = this->active_group.getObjName() + "/" + _dataset_name;
                    H5::Exception::dontPrint();
                    if (!this->datasets.contains(dataset_name_)) {
                        throw H5::DataSetIException("WriteDataset", "Cannot find dataset [" + dataset_name_ + "]\n");
                    }

                    H5::DataSet dataset_ = this->datasets.at(dataset_name_);
                    dataset_.write(data, TypeData);

                }
                catch (H5::DataSetIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
                catch (H5::FileIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief check if dataset exists, if you want to load and check the datasets exists don't forget to call GetAllData() first
             * @param _dataset_name
             * @return bool
             */
            bool CheckDataSetExists(const std::string& _dataset_name){

                return this->datasets.contains(_dataset_name);

            }

            /**
             * @brief check if attribute exists, if you want to load and check the attr exists don't forget to call GetAllData() first
             * @param _attr_name
             * @return bool
             */
            bool CheckAttributeExists(const std::string& _attr_name) {

                return this->attributes.contains(_attr_name);

            }

            /**
             * @brief Check if current file was load into class
             * @return bool
             */
            bool CheckFileExists(){
                return this->file.isValid(this->file.getId());
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
                    std::string dataset_name_ = this->active_group.getObjName() + "/" + _dataset_name;
                    std::string attr_name_ = dataset_name_ + "/" + _attr_name;
                    
                    if(this->attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] already exists\n");
                    }
                    
                    if(!this->datasets.contains(dataset_name_)){
                        throw H5::AttributeIException("CreateAttribute","Cannot find dataset ["+dataset_name_+"]\n");
                    }

                    if(!this->dataspaces.contains(_dataspace_name)){
                        throw H5::AttributeIException("CreateAttribute","Cannot find dataspace ["+_dataspace_name+"]\n");
                    }

                    H5::DataSet dataset_ = this->datasets.at(dataset_name_);
                    H5::DataSpace dataspace_ = this->dataspaces.at(_dataspace_name);
                    
                    this->attributes.insert({
                        attr_name_,
                        dataset_.createAttribute(_attr_name,_datatype,dataspace_)
                    });

                }catch(H5::AttributeIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::DataSpaceIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch (H5::DataSetIException& err) {
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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
                    
                    if(this->attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] already exists\n");
                    }

                    if(!this->groups.contains(_group_name)){
                        throw H5::AttributeIException("CreateAttribute","Group ["+_group_name+"] does not exists\n");
                    }

                    if(!this->dataspaces.contains(_dataspace_name)){
                        throw H5::AttributeIException("CreateAttribute","Cannot find dataspace ["+_dataspace_name+"]\n");
                    }

                    H5::Group group_ = this->groups.at(_group_name);
                    H5::DataSpace dataspace_ = this->dataspaces.at(_dataspace_name);
                    
                    this->attributes.insert({
                        attr_name_,
                        group_.createAttribute(_attr_name,_datatype,dataspace_)
                    });

                    group_.close();

                }catch(H5::AttributeIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::DataSpaceIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch (H5::DataSetIException& err) {
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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
                    std::string dataset_name_ = this->active_group.getObjName() + "/" + _dataset_name;
                    std::string attr_name_ = dataset_name_ + "/" + _attr_name;

                    if(!this->attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }

                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.write(_datatype, _data);

                }catch(H5::AttributeIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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

                    if(!this->attributes.contains(attr_name_)){
                        throw H5::AttributeIException("CreateAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }

                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.write(_datatype, _data);

                }catch(H5::AttributeIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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

                    if(!this->attributes.contains(attr_name_)){
                        throw H5::DataSetIException("ReadAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }
                    
                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.read(_datatype, _data);

                }catch(H5::DataSetIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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

                    if (!this->attributes.contains(attr_name_)) {
                        throw H5::DataSetIException("ReadAttribute", "Attribute [" + attr_name_ + "] does not exists\n");
                    }

                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.read(_datatype, _data);

                }
                catch (H5::DataSetIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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

                    if (!this->attributes.contains(attr_name_)) {
                        throw H5::DataSetIException("ReadAttribute", "Attribute [" + attr_name_ + "] does not exists\n");
                    }

                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.read(_datatype,_data);

                }
                catch (H5::AttributeIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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

                    if (!this->attributes.contains(attr_name_)) {
                        throw H5::DataSetIException("ReadAttribute", "Attribute [" + attr_name_ + "] does not exists\n");
                    }

                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.read(_datatype, _data);

                }
                catch (H5::AttributeIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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
                    if (!this->datasets.contains(_dataset_name)) {
                        throw H5::DataSetIException("ReadDataset", "Cannot find dataset [" + _dataset_name + "]\n");
                    }

                    H5::DataSet dataset_ = this->datasets.at(_dataset_name);
                    dataset_.read(data, TypeData);

                }
                catch (H5::DataSetIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
                catch (H5::FileIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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
                    if (!this->datasets.contains(_dataset_name)) {
                        throw H5::DataSetIException("ReadDataset", "Cannot find dataset [" + _dataset_name + "]\n");
                    }

                    H5::DataSet dataset_ = this->datasets.at(_dataset_name);
                    dataset_.read(data, TypeData);

                }
                catch (H5::DataSetIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
                catch (H5::FileIException& err) {
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Check if group exists
             * @param _group_name 
             * @return bool
             */
            bool CheckGroupExists(const std::string& _group_name) {
               
                if (!this->groups.contains(_group_name)) {
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

                    if(this->groups.contains(_group_name)){
                        throw H5::GroupIException("CreateGroup","Group ["+_group_name+"] already exists\n");
                    }

                    H5::Group group_ = this->active_group.createGroup(_group_name);
                    this->groups.insert({
                        _group_name,
                        group_
                    });

                }catch(H5::FileIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::GroupIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
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

                    if(!this->groups.contains(_target_group_name)){
                        throw H5::GroupIException("CreateGroup","Target group ["+_group_name+"] does not exists\n");
                    }
                    if(this->groups.contains(_group_name)){
                        throw H5::GroupIException("CreateGroup","Group ["+_group_name+"] already exists\n");
                    }

                    H5::Group group_ = this->groups.at(_target_group_name);
                    H5::Group new_group_ = group_.createGroup(_group_name);
                    this->groups.insert({
                        _group_name,
                        group_
                    });

                }catch(H5::FileIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::GroupIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }

            }



        };

    }

}