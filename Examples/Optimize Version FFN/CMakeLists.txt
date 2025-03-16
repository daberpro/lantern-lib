cmake_minimum_required(VERSION 3.31)
project(OptimizeVersionFFN)

add_executable(XOR XOR.cpp)
target_link_libraries(XOR PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork)