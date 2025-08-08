cmake_minimum_required(VERSION 3.31)
project(lantern VERSION 0.0.1)

option(COPY_DATASET "Copy default dataset (Mnist and other) folder to build output" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_PREFIX_PATH D:/module)

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(MatPlot++ REQUIRED)

file(
    GLOB HEADER_FILES 
    "${CMAKE_SOURCE_DIR}/ConvolutionalNeuralNetwork/*.h"
    "${CMAKE_SOURCE_DIR}/FeedForwardNetwork/*.h"
    "${CMAKE_SOURCE_DIR}/Headers/*.h"
    "${CMAKE_SOURCE_DIR}/Dataset/*.h"
)

# add_definitions(/MTd)
add_subdirectory(${CMAKE_SOURCE_DIR}/AutoGradient)
add_subdirectory(${CMAKE_SOURCE_DIR}/Examples)
add_executable(${PROJECT_NAME} main.cpp pch.h ${HEADER_FILES})
target_compile_definitions(${PROJECT_NAME} PUBLIC H5_BUILT_AS_STATIC_LIB)


target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient ArrayFire::afcpu Matplot++::nodesoup Matplot++::matplot hdf5_cpp-static)
target_precompile_headers(${PROJECT_NAME} PUBLIC pch.h)

# if(COPY_DATASET)
#     set(DATASET_DIR "${CMAKE_SOURCE_DIR}/Dataset/mnist")
#     set(DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/dataset")

#     add_custom_target(copy_dataset ALL
#         COMMAND ${CMAKE_COMMAND} -E echo "Copying dataset..."
#         COMMAND ${CMAKE_COMMAND} -E copy_directory
#                 "${DATASET_DIR}"
#                 "${DEST_DIR}"
#         COMMENT "Copying MNIST dataset to build folder"
#         VERBATIM
#     )

#     # Make your example project depend on it
#     add_dependencies(${PROJECT_NAME} copy_dataset)
# endif()