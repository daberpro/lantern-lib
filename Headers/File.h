#pragma once
#include "../pch.h"
#include <H5Cpp.h>

namespace lantern {

    namespace file {

        class LanternHDF5 {
        private:

            H5std_string filename;
            H5::H5File file;
            H5::Group active_group;
            std::unordered_map<std::string,H5::DataSet> datasets;
            std::unordered_map<std::string,H5::DataSpace> dataspaces;
            std::unordered_map<std::string,H5::Group> groups;
            std::unordered_map<std::string,H5::Attribute> attributes;

            void GetAllDataInfo(const H5::Group& root_group) {
                std::vector<H5::Group> groups_stack = {root_group};
                std::vector<std::string> current_path_stack = {""};
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
                        this->attributes.insert({ current_path + "/" + current_group.getObjName() + "/" + attr.getName(), attr});
                    }
                }
            }
            
            
        public:

            /**
             * @brief Set active group, the active group is a group which will process all opeartion on the class
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
            
            /**
             * @brief Load file from the filename
             */
            void LoadFile(std::string _filename,uint32_t AvailableAction){
                try{
                    this->file = H5::H5File(_filename,AvailableAction);
                }catch(H5::FileIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Get dataset dimension by dataset name
             */
            lantern::utility::Vector<hsize_t> GetDatasetDims(const std::string& _dataset_name){
                lantern::utility::Vector<hsize_t> data;
                if(!this->datasets.contains(_dataset_name)){
                    std::cout << "Dataset ["+_dataset_name+"] not found\n";
                    data.clear();
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
             */
            lantern::utility::Vector<hsize_t> GetAttrDims(const std::string& _attr_name){
                lantern::utility::Vector<hsize_t> data;
                if(!this->attributes.contains(_attr_name)){
                    std::cout << "Attr ["+_attr_name+"] not found\n";
                    data.clear();
                    return data;   
                }
                H5::Attribute& attribute = this->attributes.at(_attr_name);
                H5::DataSpace dataspace = attribute.getSpace();
                uint32_t rank = dataspace.getSimpleExtentNdims();
                data = lantern::utility::Vector<hsize_t>(rank);
                dataspace.getSimpleExtentDims(data.getData());
                data.explicitTotalItem(rank);
                return data;
            }

            /**
             * @brief Print out dataset dimension
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
             */
            std::unordered_map<std::string, H5::DataSpace>& GetDataSpaces(){
                return this->dataspaces;
            }

            /**
             * @brief Create new dataspace
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
             * @brief Write dataset, warning this only works if dataset already create or load using GetAllData()
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
             * @brief check if dataset exists, if you want to load and check the datasets exists don't forget to call GetAllData() first
             */
            bool CheckDataSetExists(const std::string& _dataset_name){

                return this->datasets.contains(_dataset_name);

            }

            /**
             * @brief Check if current file was load into class
             */
            bool CheckFileExists(){
                return this->file.isValid(this->file.getId());
            }

            /**
             * @brief Create new attribute at dataset
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
             */
            template <typename DataType, typename Data>
            void ReadAttribute(
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
                        throw H5::DataSetIException("ReadAttribute","Attribute ["+attr_name_+"] does not exists\n");
                    }
                    
                    H5::Attribute attr_ = this->attributes.at(attr_name_);
                    attr_.write(_datatype, _data);
                    attr_.close();

                }catch(H5::AttributeIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }
            
            /**
             * @brief Read exisiting attribute, warning this only works if attribute already create or load using GetAllData()
             */
            template <typename Data>
            void ReadDataset(const std::string& _dataset_name, Data* data, const H5::PredType& TypeData){
                try{
                    
                    H5::Exception::dontPrint();
                    if(!this->datasets.contains(_dataset_name)){
                        throw H5::DataSetIException("ReadDataset","Cannot find dataset ["+_dataset_name+"]\n");
                    }

                    H5::DataSet dataset_ =  this->datasets.at(_dataset_name);
                    dataset_.read(data, TypeData);  
                    dataset_.close();                      

                }catch(H5::DataSetIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::FileIException& err){
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }
            }

            /**
             * @brief Create new group
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
             * @brief Create new group
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

            ~LanternHDF5(){

                try{
                    H5::Exception::dontPrint();
                    for(auto& [d_key,d_value] : this->datasets){
                        d_value.close();
                    }
                    for(auto& [a_key,a_value] : this->attributes){
                        a_value.close();
                    }
                    for(auto& [g_key,g_value] : this->groups){
                        g_value.close();
                    }
                    this->file.close();
                }catch(H5::FileIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::AttributeIException& err){
                    err.printErrorStack();
                    std::cout << err.getDetailMsg() << '\n';
                    exit(EXIT_FAILURE);
                }catch(H5::DataSetIException& err){
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