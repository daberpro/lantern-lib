find_package(GTest REQUIRED)
# find_package(ArrayFire REQUIRED)

add_library(test_lib INTERFACE)
target_link_libraries(test_lib INTERFACE GTest::gtest AutoGradient)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/AutoGradient)
