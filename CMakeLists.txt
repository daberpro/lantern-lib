cmake_minimum_required(VERSION 3.31)
project(lantern CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_PREFIX_PATH "D:/module2;D:/v3")

add_definitions(/MTd)
add_subdirectory(${CMAKE_SOURCE_DIR}/AutoGradient)
add_subdirectory(${CMAKE_SOURCE_DIR}/FeedForwardNetwork)
# add_subdirectory(${CMAKE_SOURCE_DIR}/test)
add_executable(${PROJECT_NAME} main.cpp pch.h)

find_package(ArrayFire REQUIRED)
target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR}/Headers
    ${PROJECT_SOURCE_DIR}/AutoGradient
    ${PROJECT_SOURCE_DIR}/FeedForwardNetwork
)
target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient FeedForwardNetwork ArrayFire::afcpu)
target_precompile_headers(${PROJECT_NAME} PUBLIC pch.h)