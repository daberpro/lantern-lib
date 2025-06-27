cmake_minimum_required(VERSION 3.31)
project(lantern VERSION 0.0.1)

option(COPY_DATASET "Copy default dataset (Mnist and other) folder to build output" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_PREFIX_PATH D:/module)

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(MatPlot++ REQUIRED)

# add_definitions(/MTd)
add_subdirectory(${CMAKE_SOURCE_DIR}/AutoGradient)
add_subdirectory(${CMAKE_SOURCE_DIR}/Examples)
add_executable(${PROJECT_NAME} main.cpp pch.h)
target_compile_definitions(${PROJECT_NAME} PUBLIC H5_BUILT_AS_STATIC_LIB)

target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR}/Headers
    ${PROJECT_SOURCE_DIR}/Dataset
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient ArrayFire::afcpu Matplot++::nodesoup Matplot++::matplot hdf5_cpp-static)
target_precompile_headers(${PROJECT_NAME} PUBLIC pch.h)

if(COPY_DATASET)
    set(DATASET mnist)
    foreach(dataset ${DATASET})    
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/Dataset/${dataset}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/dataset"
        )
    endforeach()
endif()