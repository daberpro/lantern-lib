cmake_minimum_required(VERSION 3.31)
project(Examples)

add_executable(XORAdvance XORAdvance.cpp)
target_link_libraries(XORAdvance PRIVATE ArrayFire::afcpu hdf5_cpp-static)

add_executable(XORSimple XORSimple.cpp)
target_link_libraries(XORSimple PRIVATE ArrayFire::afcpu hdf5_cpp-static)

add_executable(MultipleClass MultipleClass.cpp)
target_link_libraries(MultipleClass PRIVATE ArrayFire::afcpu hdf5_cpp-static)

add_executable(MultipleClassSimple MultipleClassSimple.cpp)
target_link_libraries(MultipleClassSimple PRIVATE ArrayFire::afcpu hdf5_cpp-static)

add_executable(MnistDataset MnistDataset.cpp)
target_link_libraries(MnistDataset PRIVATE ArrayFire::afcpu)

if(COPY_DATASET)
    set(DATASET mnist)
    foreach(dataset ${DATASET})
        foreach(target XORAdvance XORSimple MultipleClass MultipleClassSimple)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/Dataset/${dataset}"
                "$<TARGET_FILE_DIR:${target}>/dataset"
            )
        endforeach()
    endforeach()
endif()