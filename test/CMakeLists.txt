add_executable(test_convolution
	test_convolution.cpp
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

target_include_directories(test_convolution
PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/../deps/thirdparty/googletest/googletest
	${CMAKE_CURRENT_SOURCE_DIR}/../deps/thirdparty/googletest/googletest/include
)

target_link_libraries(test_convolution
	gtest
	gtest_main
	convolution
)