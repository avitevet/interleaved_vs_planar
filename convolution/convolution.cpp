#include "convolution.h"

bool transposePlanar(const std::vector<float>& src, unsigned int height, unsigned int width, unsigned int numChannels, std::vector<float>& dst) {
	// size sanity checks
	if (src.size() > dst.size()) {
		return false;
	}

	if (height * width * numChannels > src.size()) {
		return false;
	}

	const unsigned int srcRowStride = width;
	const unsigned int dstRowStride = height;

	for (unsigned int ch = 0; ch < numChannels; ch++) {
		const unsigned int chStart = ch * height * width;

		for (unsigned int srcRow = 0; srcRow < height; srcRow++) {
			const unsigned int srcRowStart = chStart + srcRow * srcRowStride;

			for (unsigned int srcCol = 0; srcCol < width; srcCol++) {
				dst[chStart + srcCol * dstRowStride + srcRow] = src[srcRowStart + srcCol];
			}
		}
	}

	return true;
}