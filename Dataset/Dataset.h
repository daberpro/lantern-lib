#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"
#include "../Headers/DataProcessing.h"
#include "../Headers/File.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../Headers/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../Headers/stb_image_resize2.h"
#include "Idxd.h"

/**
 * @defgroup LanternDataset Dataste utility class and function to manage dataset
 */

namespace lantern {

    namespace dataset {

        /**
         * @brief width and height wrapper for image from MnistDataset
         * @ingroup LanternDataset
         */
        struct MnistImageSize{
            int m_width;
            int m_height;

            MnistImageSize(const int& _width, const int& _height): m_width(_width), m_height(_height){}
        };

        /**
         * @brief Mnist dataset wrapper
         * @ingroup LanternDataset
         */
        class MnistDataset {
        private:

            std::string m_path, m_current_path;
            lantern::idxd::Idx<uint8_t> m_train_images;
            lantern::idxd::Idx<uint8_t> m_train_labels;
            lantern::idxd::Idx<uint8_t> m_images;
            lantern::idxd::Idx<uint8_t> m_labels;

        public:

            MnistDataset() {
                this->m_current_path = std::filesystem::current_path().string();
                std::string dataset_dir_ = this->m_current_path + "/dataset/";
                this->m_train_images = lantern::idxd::Idx<uint8_t>(dataset_dir_ + "train-images.idx3-ubyte");
                this->m_train_labels = lantern::idxd::Idx<uint8_t>(dataset_dir_ + "train-labels.idx1-ubyte");
                this->m_images = lantern::idxd::Idx<uint8_t>(dataset_dir_ + "t10k-images.idx3-ubyte");
                this->m_labels = lantern::idxd::Idx<uint8_t>(dataset_dir_ + "t10k-labels.idx1-ubyte");
            }

            /**
             * @brief Print label and image train at index
             * @param index 
             */
            void PrintTrainDataAt(const uint32_t& _index){
                auto image_ = this->m_train_images[static_cast<long long>(_index)];
                auto* dims_ = this->m_train_images.GetDims();
                for(uint32_t row = 0; row < dims_->at(1); row++){
                    for(uint32_t col = 0; col < dims_->at(2); col++){
                        std::cout << (image_[row * dims_->at(1) + col] > 122? "#" : ".") << " ";
                    }
                    std::cout << '\n';
                }
                std::cout << "Label : " << static_cast<int>(*this->m_train_labels[_index]) << '\n';
            }

            /**
             * @brief Print image and label test at index
             * @param index 
             */
            void PrintDataAt(const uint32_t& _index){
                auto image_ = this->m_images[static_cast<long long>(_index)];
                auto* dims_ = this->m_images.GetDims();
                for(uint32_t row = 0; row < dims_->at(1); row++){
                    for(uint32_t col = 0; col < dims_->at(2); col++){
                        std::cout << (image_[row * dims_->at(1) + col] > 122? "#" : ".") << " ";
                    }
                    std::cout << '\n';
                }
                std::cout << "Label : " << static_cast<int>(*this->m_labels[_index]) << '\n';
            }

            /**
             * @brief Get train image at index
             * @param index
             * @return uint8_t*
             */
            uint8_t* GetTrainImageAt(const long long& _index){
               return this->m_train_images[_index];
            }

            /**
             * @brief Get train label data at index, please make sure to cast the result using int
             * @param index
             * @return uint8_t*
             */
            uint8_t* GetTrainLabelAt(const long long& _index){
               return this->m_train_labels[_index];
            }

            /**
             * @brief Get test image at index 
             * @param index 
             * @return uint8_t*
             */
            uint8_t* GetImageAt(const long long& _index){
               return this->m_images[_index];
            }

            /**
             * @brief Get test label data at index, please make sure to cast the result using int 
             * @param index 
             * @return uint8_t*
             */
            uint8_t* GetLabelAt(const long long& _index){
               return this->m_labels[_index];
            }

            /**
             * @brief Get dimension of train image
             * @return lantern::utility::Vector<int>*
             */
            lantern::utility::Vector<int>* GetTrainImageDims(){
                return this->m_train_images.GetDims();
            }

            /**
             * @brief Get dimension of train label
             * @return lantern::utility::Vector<int>*
             */
            lantern::utility::Vector<int>* GetTrainLabelDims(){
                return this->m_train_labels.GetDims();
            }

            /**
             * @brief Get dimension of test image
             * @return lantern::utility::Vector<int>*
             */
            lantern::utility::Vector<int>* GetImageDims(){
                return this->m_images.GetDims();
            }
            
            /**
             * @brief Get dimension of train label
             * @return lantern::utility::Vector<int>*
             */
            lantern::utility::Vector<int>* GetLabelDims(){
                return this->m_train_labels.GetDims();
            }

            /**
             * @brief Get images data form train images
             * @return lantern::utility::Vector<uint8_t>*
             */
            lantern::utility::Vector<uint8_t>* GetTrainImagesData(){
                return this->m_train_images.GetData();
            }

            /**
             * @brief Get images data from test images
             * @return lantern::utility::Vector<uint8_t>*
             */
            lantern::utility::Vector<uint8_t>* GetImagesData(){
                return this->m_images.GetData();
            }

            /**
             * @brief Get data from train labels
             * @return lantern::utility::Vector<uint8_t>*
             */
            lantern::utility::Vector<uint8_t>* GetTrainLabelsData(){
                return this->m_train_labels.GetData();
            }

            /**
             * @brief Get data from test labels
             * @return lantern::utility::Vector<uint8_t>*
             */
            lantern::utility::Vector<uint8_t>* GetLabelsData(){
                return this->m_labels.GetData();
            }

            /**
             * @brief Get total images from train images
             * @return uint32_t
             */
            uint32_t GetTotalTrainImages(){
                auto* dims_ = this->m_train_images.GetDims();
                return (*dims_)[0]; 
            }

            /**
             * @brief Get total labels from train labels
             * @return uint32_t
             */
            uint32_t GetTotalTrainLabels(){
                auto* dims_ = this->m_train_labels.GetDims();
                return (*dims_)[0]; 
            }

            /**
             * @brief Get total labels from test labels
             * @return uint32_t
             */
            uint32_t GetTotalLabels(){
                auto* dims_ = this->m_labels.GetDims();
                return (*dims_)[0]; 
            }

            /**
             * @brief Get train image sizes
             * @return lantern::dataset::MnistImageSize
             */
            MnistImageSize GetTrainImageSizes(){
                auto* dims_ = this->m_train_images.GetDims();
                return MnistImageSize((*dims_)[1], (*dims_)[2]);
            }

            /**
             * @brief Get test image sizes
             * @return lantern::dataset::MnistImageSize
             */
            MnistImageSize GetImageSizes(){
                auto* dims_ = this->m_images.GetDims();
                return MnistImageSize((*dims_)[1], (*dims_)[2]);
            }

            /**
             * @brief Get path 
             * @return std::string_view
             */
            std::string_view GetPath(){
                return this->m_path;
            }
        
        };

        
        template <uint32_t TOTAL_IMAGES, uint32_t IMG_WIDTH, uint32_t IMG_HEIGHT, bool IsColor>
        class ImageLoader
        {
        private:
            std::mutex m_mutex;
            std::condition_variable m_producer, m_consumer;

            lantern::utility::Vector<uint32_t> m_each_class_sizes;
            std::unordered_map<std::string, lantern::utility::Vector<std::string>> m_image_paths;
            std::unordered_map<std::string, lantern::utility::Vector<uint8_t>> m_image_cache;
            std::unordered_map<std::string, std::array<std::string,TOTAL_IMAGES>> m_label_cache;
            std::unordered_map<std::string, lantern::file::CSVFile> m_labels;
            std::string m_active_dataset;
            size_t m_allocation = TOTAL_IMAGES * IMG_WIDTH * IMG_HEIGHT * (IsColor ? 3 : 1);
            std::thread m_thread_loader;
            std::atomic<bool> m_stop_thread = false;

            uint32_t m_head = 0, m_tail = 0, m_count = 0;

            void Put(const std::string &_image_path)
            {
                std::unique_lock<std::mutex> lock_(this->m_mutex);
                this->m_producer.wait(lock_, [this](){ return this->m_count < TOTAL_IMAGES || this->m_stop_thread; });
                if (this->m_stop_thread)
                {
                    return;
                }
                this->m_CheckDatasetValid();
                auto &image_data_ = this->m_image_cache[this->m_active_dataset];
                auto &label_data_ = this->m_label_cache[this->m_active_dataset];
                int width_, height_, channels_;
                stbir_pixel_layout layout_ = IsColor? STBIR_RGB : STBIR_1CHANNEL;
                uint8_t* image_ = stbi_load(_image_path.c_str(), &width_, &height_, &channels_, IsColor ? 3 : 1);
                if (!image_) {
                    std::println("Error ImageLoader, STB cannot load image \"{}\" because {}", _image_path, stbi_failure_reason());
                    return;
                }
                try {
                    stbir_resize_uint8_linear(
                        image_,
                        width_, height_, 0,
                        image_data_.data() + (size_t)(this->m_tail * IMG_WIDTH * IMG_HEIGHT * (IsColor ? 3 : 1)),
                        IMG_WIDTH, IMG_HEIGHT, 0,
                        layout_
                    );
                    
                    label_data_[this->m_tail] = std::filesystem::path(_image_path).parent_path().filename().string();

                } catch (...) {
                    stbi_image_free(image_);
                    return;
                }
                stbi_image_free(image_);
                this->m_tail = (this->m_tail + 1) % TOTAL_IMAGES;
                this->m_count++;
                this->m_consumer.notify_all();
            }

            void Loaders()
            {

                this->m_each_class_sizes.back() -= 1;
                lantern::utility::Vector<uint32_t> batch_index_;
                uint32_t total_size_of_class_ = 0;
                for (auto &size_ : this->m_each_class_sizes)
                {
                    total_size_of_class_ += size_;
                }
                if (total_size_of_class_ == 0)
                {
                    throw std::runtime_error("Error ImageLoader, No image found in dataset");
                }

                while (!this->m_stop_thread)
                {
                    auto &image_paths_ = this->m_image_paths[this->m_active_dataset];
                    lantern::data::GetRandomSampleClassIndex<TOTAL_IMAGES>(batch_index_, this->m_each_class_sizes, total_size_of_class_);
                    for (auto &index : batch_index_)
                    {
                        if (this->m_stop_thread)
                        {
                            return;
                        }
                        this->m_Put(image_paths_[index]);
                    }
                }
            }

            std::unordered_set<std::string> extension_accepted = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
            };
            /**
             * @brief Check if the file extension was an image or not
             * @param path 
             * @return bool
             */
            bool IsImage(const std::filesystem::path& _path){

                std::string ext_ = _path.extension().string();
                std::transform(ext_.begin(), ext_.end(), ext_.begin(), [](unsigned char c) { return std::tolower(c); });
                return this->m_extension_accepted.find(ext_) != this->m_extension_accepted.end();

            }

        public:
            ImageLoader() = default;

            uint8_t *Get()
            {
                std::unique_lock<std::mutex> lock_(this->m_mutex);
                this->m_consumer.wait(lock_, [this](){ return this->m_count > 0 || this->m_stop_thread; });
                if (this->m_stop_thread || this->m_count == 0)
                {
                    return nullptr; // Stop the thread if requested
                }
                auto &image_data_ = this->m_image_cache[this->m_active_dataset];
                uint8_t *image_ = image_data_.data() + (size_t)(this->m_head * IMG_WIDTH * IMG_HEIGHT * (IsColor ? 3 : 1));
                this->m_head = (this->m_head + 1) % TOTAL_IMAGES;
                this->m_count--;
                this->m_producer.notify_all();
                return image_;
            }

            void CheckDatasetValid()
            {
                if (this->m_active_dataset.empty())
                {
                    throw std::runtime_error("Error ImageLoader, No dataset selected");
                }
            }

            void GetImagesDataFromFolder(const std::filesystem::path &_path)
            {
                this->m_CheckDatasetValid();
                if (std::filesystem::exists(_path) && std::filesystem::is_directory(_path))
                {
                    auto &image_paths_ = this->m_image_paths[this->m_active_dataset];
                    uint32_t class_size_ = 0;
                    for (auto &file_ : std::filesystem::directory_iterator(_path))
                    {
                        if (this->m_IsImage(file_))
                        {
                            image_paths_.push_back(file_.path().string());
                            class_size_++;
                        };
                    }
                    this->m_each_class_sizes.push_back(class_size_);
                }
                else
                {
                    throw std::runtime_error(std::format("Error ImageLoader, cannot access folder path \"{}\" looks like deleted or moved", _path.string()));
                }
            }

            void SelectDatasetToModify(const std::string &_dataset_name)
            {
                if (!this->m_image_cache.contains(_dataset_name))
                {
                    throw std::runtime_error(std::format("Error ImageLoader, dataset \"{}\" do not exists", _dataset_name));
                }
                this->m_active_dataset = _dataset_name;
            }

            void CreateDatasetForFolder(const std::string &_dataset_name)
            {
                if (this->m_image_cache.contains(_dataset_name))
                {
                    throw std::runtime_error(std::format("Error ImageLoader, Cannot create dataset \"{}\" because already exists", _dataset_name));
                }
                this->m_image_cache[_dataset_name] = lantern::utility::Vector<uint8_t>(this->m_allocation);
            }

            void Run()
            {
                this->m_thread_loader = std::thread(&ImageLoader::Loaders, this);
            }

            void Stop()
            {
                {
                    std::lock_guard<std::mutex> lock_(this->m_mutex);
                    this->m_stop_thread = true;
                }
                this->m_producer.notify_all(); // Notify the producer to stop waiting
                this->m_consumer.notify_all();
                this->m_thread_loader.join();
            }

            /**
             * @brief Get CSV file from folder
             * @param _path 
             */
            void ReadCSVLabelDataFromFolder(const std::filesystem::path& _path) {
                this->m_CheckDatasetValid();
                this->m_labels[this->m_active_dataset] = lantern::file::ReadCSVFile(_path);
            }

            template <af::dtype Type = f32>
            void GetAsAF(af::array &_img){
                this->m_CheckDatasetValid();
                af::array flat_(IMG_HEIGHT * IMG_WIDTH* 3, this->m_Get());
                af::array R_ = af::moddims(flat_(af::seq(0, af::end, 3)), IMG_HEIGHT, IMG_WIDTH);
                af::array G_ = af::moddims(flat_(af::seq(1, af::end, 3)), IMG_HEIGHT, IMG_WIDTH);
                af::array B_ = af::moddims(flat_(af::seq(2, af::end, 3)), IMG_HEIGHT, IMG_WIDTH);

                _img = af::join(2, R_, G_, B_);
                _img = af::reorder(_img, 1, 0, 2);
                _img = _img.as(Type) / 255;
            }

            template <af::dtype Type = f32>
            void GetAsAF(af::array &_img, std::string &_label){
                this->m_CheckDatasetValid();
                af::array flat_(IMG_HEIGHT * IMG_WIDTH* 3, this->m_Get());
                af::array R_ = af::moddims(flat_(af::seq(0, af::end, 3)), IMG_HEIGHT, IMG_WIDTH);
                af::array G_ = af::moddims(flat_(af::seq(1, af::end, 3)), IMG_HEIGHT, IMG_WIDTH);
                af::array B_ = af::moddims(flat_(af::seq(2, af::end, 3)), IMG_HEIGHT, IMG_WIDTH);

                _img = af::join(2, R_, G_, B_);
                _img = af::reorder(_img, 1, 0, 2);
                _img = _img.as(Type) / 255.0f;

                auto& label_data_ = this->m_label_cache[this->m_active_dataset];
                _label = label_data_.at(this->m_head > 0? this->m_head - 1 : TOTAL_IMAGES - 1);
            }

            template <typename T>
            auto GetCSVLabelAtRow(const uint32_t& _row){
                this->m_CheckDatasetValid();
                return this->m_labels[this->m_active_dataset].Row<T>(_row);
            }

            template <typename T>
            auto GetCSVLabelAtCol(const uint32_t& _col){
                this->m_CheckDatasetValid();
                return this->m_labels[this->m_active_dataset].Col<T>(_col);
            }

            auto& GetCSVFile(){
                this->m_CheckDatasetValid();
                return this->m_labels[this->m_active_dataset];
            }

        };
      

        template <uint32_t TOTAL_IMAGES, uint32_t IMG_WIDTH, uint32_t IMG_HEIGHT, bool IsColor>
        class ImageLoaderGuard {
        private:
            using Type = ImageLoader<TOTAL_IMAGES, IMG_WIDTH, IMG_HEIGHT, IsColor>;
            Type* imageLoader = nullptr;

        public:

            ImageLoaderGuard(Type* imageLoader) : imageLoader(imageLoader) {
                if (this->m_imageLoader == nullptr) {
                    throw std::runtime_error("Error ImageLoaderGuard, The pointer to image loader canno be nullptr");
                }
            }

            void Run() {
                this->m_imageLoader->Run();
            }

            ~ImageLoaderGuard() {
                std::println("Image Guard Stopped");
                this->m_imageLoader->Stop();
            }

        };


    }

}