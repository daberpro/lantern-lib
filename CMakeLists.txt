cmake_minimum_required(VERSION 3.31)
project(lantern CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_PREFIX_PATH "D:/module/v3")

find_package(ArrayFire REQUIRED)

add_definitions(/MTd)
add_subdirectory(${CMAKE_SOURCE_DIR}/ExternalDependencies/matplotplusplus)
add_subdirectory(${CMAKE_SOURCE_DIR}/AutoGradient)
# add_subdirectory(${CMAKE_SOURCE_DIR}/FeedForwardNetwork)
add_subdirectory(${CMAKE_SOURCE_DIR}/Examples)
add_executable(${PROJECT_NAME} main.cpp pch.h)

target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR}/Headers
    ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source
)
# target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient FeedForwardNetwork ArrayFire::afcpu matplot)
target_link_libraries(${PROJECT_NAME} PUBLIC AutoGradient ArrayFire::afcuda matplot)
target_precompile_headers(${PROJECT_NAME} PUBLIC pch.h)