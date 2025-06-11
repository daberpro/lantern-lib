cmake_minimum_required(VERSION 3.31)
project(Examples)

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(XORAdvance XORAdvance.cpp)
target_include_directories(XORAdvance PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source ${HDF5_INCLUDE_DIRS})
target_link_libraries(XORAdvance PRIVATE ArrayFire::afcpu AutoGradient matplot hdf5_cpp-static)

add_executable(MultipleClass MultipleClass.cpp)
target_include_directories(MultipleClass PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source ${HDF5_INCLUDE_DIRS})
target_link_libraries(MultipleClass PRIVATE ArrayFire::afcpu AutoGradient matplot hdf5_cpp-static)

add_executable(MultipleClassSimple MultipleClassSimple.cpp)
target_include_directories(MultipleClassSimple PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source ${HDF5_INCLUDE_DIRS})
target_link_libraries(MultipleClassSimple PRIVATE ArrayFire::afcpu AutoGradient matplot hdf5_cpp-static)

add_executable(GenderByWeightAndHeightSimple GenderByWeightAndHeightSimple.cpp)
target_include_directories(GenderByWeightAndHeightSimple PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source ${HDF5_INCLUDE_DIRS})
target_link_libraries(GenderByWeightAndHeightSimple PRIVATE ArrayFire::afcpu AutoGradient matplot hdf5_cpp-static)