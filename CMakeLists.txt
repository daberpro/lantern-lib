cmake_minimum_required(VERSION 3.31)
project(lantern VERSION 0.0.1)

option(COPY_DATASET "Copy default dataset (Mnist and other) folder to build output" ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(CMAKE_CXX_STANDARD 23)
list(APPEND CMAKE_PREFIX_PATH "D:/module")
include_directories("D:/module/include")

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED)
find_package(ArrayFire REQUIRED)

file(
    GLOB HEADER_FILES 
    "${CMAKE_SOURCE_DIR}/ConvolutionalNeuralNetwork/*.h"
    "${CMAKE_SOURCE_DIR}/FeedForwardNetwork/*.h"
    "${CMAKE_SOURCE_DIR}/Headers/*.h"
    "${CMAKE_SOURCE_DIR}/Dataset/*.h"
    "${CMAKE_SOURCE_DIR}/Base/*.h"
)


# add_subdirectory(${CMAKE_SOURCE_DIR}/AutoGradient)
add_subdirectory(${CMAKE_SOURCE_DIR}/Examples)
add_executable(${PROJECT_NAME} main.cpp pch.h ${HEADER_FILES})

# only works on visual studio for debugging
if(MSVC)
  target_compile_options(${PROJECT_NAME} PUBLIC "/ZI")
  target_link_options(${PROJECT_NAME} PUBLIC "/INCREMENTAL")
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC H5_BUILT_AS_STATIC_LIB)

target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${HDF5_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_PREFIX_PATH}/lib)
target_link_libraries(
    ${PROJECT_NAME} PUBLIC 
    # AutoGradient 
    ArrayFire::afcpu
    hdf5_cpp-static
)
target_precompile_headers(${PROJECT_NAME} PUBLIC pch.h)

if(COPY_DATASET)
    set(DATASETS
        mnist
        Iris_Flower
    )

    set(DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/dataset")

    # Create one target first
    add_custom_target(copy_dataset ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Copying datasets..."
        COMMENT "Copying datasets to build folder"
    )

    # Append copy commands for each dataset
    foreach(ds ${DATASETS})
        add_custom_target(copy_dataset_${ds}
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_SOURCE_DIR}/Dataset/${ds}"
                    "${DEST_DIR}/${ds}"
        )
        add_dependencies(copy_dataset copy_dataset_${ds})
    endforeach()

    # Make your project depend on the overall dataset copy
    add_dependencies(${PROJECT_NAME} copy_dataset)
endif()
