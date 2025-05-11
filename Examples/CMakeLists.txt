cmake_minimum_required(VERSION 3.31)
project(Examples)

add_executable(XORAdvance XORAdvance.cpp)
target_include_directories(XORAdvance PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(XORAdvance PRIVATE ArrayFire::afcpu AutoGradient matplot)

<<<<<<< HEAD
=======

add_executable(MultipleClass MultipleClass.cpp)
target_include_directories(MultipleClass PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(MultipleClass PRIVATE ArrayFire::afcpu AutoGradient matplot)

# ! DEPRECATED
# add_executable(XOR_Advance XOR_Advance.cpp)
# target_link_libraries(XOR_Advance PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork matplot)
>>>>>>> 992cc9276e65bff7d6fb8603d03962aa3937c710

add_executable(MultipleClass MultipleClass.cpp)
target_include_directories(MultipleClass PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(MultipleClass PRIVATE ArrayFire::afcpu AutoGradient matplot)


add_executable(MultipleClassSimple MultipleClassSimple.cpp)
target_include_directories(MultipleClassSimple PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(MultipleClassSimple PRIVATE ArrayFire::afcpu AutoGradient matplot)


add_executable(GenderByWeightAndHeightSimple GenderByWeightAndHeightSimple.cpp)
target_include_directories(GenderByWeightAndHeightSimple PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(GenderByWeightAndHeightSimple PRIVATE ArrayFire::afcpu AutoGradient matplot)