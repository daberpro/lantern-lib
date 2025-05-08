cmake_minimum_required(VERSION 3.31)
project(Examples)

add_executable(XORAdvance XORAdvance.cpp)
target_include_directories(XORAdvance PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(XORAdvance PRIVATE ArrayFire::afcpu AutoGradient matplot)


add_executable(MultipleClass MultipleClass.cpp)
target_include_directories(MultipleClass PUBLIC ${PROJECT_SOURCE_DIR}/ExternalDependencies/matplotplusplus/source)
target_link_libraries(MultipleClass PRIVATE ArrayFire::afcpu AutoGradient matplot)

# ! DEPRECATED
# add_executable(XOR_Advance XOR_Advance.cpp)
# target_link_libraries(XOR_Advance PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork matplot)

# ! DEPRECATED
# add_executable(XOR_Simple XOR_Simple.cpp)
# target_link_libraries(XOR_Simple PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork matplot)

# ! DEPRECATED
# add_executable(GenderByWeightAndHeight GenderByWeightAndHeight.cpp)
# target_link_libraries(GenderByWeightAndHeight PRIVATE ArrayFire::afcpu AutoGradient FeedForwardNetwork matplot)