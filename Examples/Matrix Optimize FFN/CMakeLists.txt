cmake_minimum_required(VERSION 3.31)
project(MatrixOptimizeFFN)

add_executable(XOR XOR.cpp)
add_executable(GenderByWeightAndHeight GenderByWeightAndHeight.cpp)
target_link_libraries(XOR PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork)
target_link_libraries(GenderByWeightAndHeight PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork)