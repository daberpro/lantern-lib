enable_testing()
cmake_minimum_required(VERSION 3.31)
project(AutoGradientTest CXX)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE test_lib)
add_test(NAME "Auto Gradient Test" COMMAND ${PROJECT_NAME})