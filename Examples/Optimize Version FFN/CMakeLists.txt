cmake_minimum_required(VERSION 3.31)
project(OptimizeVersionFFN)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork)