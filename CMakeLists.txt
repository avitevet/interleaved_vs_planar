cmake_minimum_required (VERSION 3.0)
project(interleaved_vs_planar)

add_subdirectory(convolution)
add_subdirectory(src)

add_subdirectory(test)
add_subdirectory(deps/thirdparty/googletest EXCLUDE_FROM_ALL)