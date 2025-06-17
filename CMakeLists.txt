cmake_minimum_required(VERSION 3.31)
project(lantern)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_PREFIX_PATH D:/module)

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED)
find_package(ArrayFire REQUIRED)

# add_definitions(/MTd)
add_subdirectory(${CMAKE_SOURCE_DIR}/ExternalDependencies/matplotplusplus)
add_subdirectory(${CMAKE_SOURCE_DIR}/AutoGradient)
add_subdirectory(${CMAKE_SOURCE_DIR}/Examples)
add_executable(${PROJECT_NAME} main.cpp pch.h)
target_compile_definitions(${PROJECT_NAME} PUBLIC H5_BUILT_AS_STATIC_LIB)

target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR}/Headers
    ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source
    ${HDF5_INCLUDE_DIRS}
)
# target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient FeedForwardNetwork ArrayFire::afcpu matplot)
target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient ArrayFire::afcpu matplot hdf5_cpp-static)
target_precompile_headers(${PROJECT_NAME} PUBLIC pch.h)