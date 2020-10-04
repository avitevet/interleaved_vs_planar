#include "convolution.h"

#include <vector>
#include <array>
#include <iostream>

void printPlanarImage(const std::vector<float>& src, unsigned int height, unsigned int width, unsigned int numChannels) {
	for (auto ch = 0U; ch < numChannels; ch++) {
		std::cout << "Channel " << ch << std::endl;

		for (auto row = 0U; row < height; row++) {
			std::cout << "{ ";
			for (auto col = 0U; col < width; col++) {
				std::cout << src[ch * height * width + row * width + col] << ", ";
			}

			std::cout << " }" << std::endl;
		}
	}
}

void printInterleavedImage(const std::vector<float>& src, unsigned int height, unsigned int width, unsigned int numChannels) {
	const unsigned int rowStride = width * numChannels;
	const unsigned int pxStride = numChannels;

	for (auto row = 0U; row < height; row++) {
		std::cout << "[ ";
		for (auto col = 0U; col < width; col++) {
			std::cout << "{ ";
			for (auto ch = 0U; ch < numChannels; ch++) {
				std::cout << src[row * rowStride + col * pxStride + ch] << ", ";
			}
			std::cout << "}, ";
		}

		std::cout << "]" << std::endl;
	}
}

int main() {
	const std::vector<float> src { 
		1.0f, 0, 2.0f, 0, 3.0f, 0, 1.0f, 0, 
		2.0f, 0, 6.0f, 0, 7.0f, 0, 2.0f, 0, 
		3.5f, 0, 2.5f, 0, 3.5f, 0, 3.5f, 0, 
		4.5f, 0, 6.5f, 0, 7.5f, 0, 4.5f, 0, 
	};

	std::vector<float> dst(src.size());

	const std::array<float, 3> blur1D{ {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f } };
	convolve1DHorizontalInterleaved(blur1D, src, 4, 4, 2, 0, dst);

	std::cout << "The source matrix is: " << std::endl;
	printInterleavedImage(src, 4, 4, 2);

	std::cout << std::endl << std::endl << "The dst matrix after horiz convolve is: " << std::endl;
	printInterleavedImage(dst, 4, 4, 2);

	std::fill(dst.begin(), dst.end(), 0.0f);

	convolve1DVerticalInterleaved(blur1D, src, 4, 4, 2, 0, dst);

	std::cout << std::endl << std::endl << "The dst matrix after vert convolve is: " << std::endl;
	printInterleavedImage(dst, 4, 4, 2);

	return 0;
}