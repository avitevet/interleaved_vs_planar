add_library(convolution
	convolution.cpp
)

target_include_directories(convolution
INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}
)
