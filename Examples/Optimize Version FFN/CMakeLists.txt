cmake_minimum_required(VERSION 3.31)
project(OptimizeVersionFFN)

add_executable(XOR_Simple_Optimizer XOR.cpp)
target_link_libraries(XOR_Simple_Optimizer PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork)