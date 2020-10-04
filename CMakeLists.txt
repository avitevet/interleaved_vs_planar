cmake_minimum_required (VERSION 3.0)
project(interleaved_vs_planar)

add_subdirectory(convolution)
add_subdirectory(test)

add_executable(interleaved_vs_planar main.cpp)

target_link_libraries(interleaved_vs_planar convolution)

