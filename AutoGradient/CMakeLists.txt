project(AutoGradient CXX)
find_package(ArrayFire REQUIRED)
add_library(${PROJECT_NAME} STATIC ReverseMode.cpp Node.cpp FeedForward.cpp)
target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} "D:/v3/include"
    ${CMAKE_SOURCE_DIR}/Headers
)
target_link_libraries(${PROJECT_NAME} PRIVATE ArrayFire::afcpu)