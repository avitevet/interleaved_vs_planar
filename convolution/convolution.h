#include <vector>

/**
* Performs 1D horizontal convolution on a single channel of an input planar format image with the given kernel.  Does not compute convolution around edge.
*
* @tparam kernelT  Kernel type - must have .size() and operator[] (std::array, std::vector, etc)
* @param[in] kernel  1D kernel to convolve with.  Must have odd length.
* @param[in] image  2D multi-channel image to convolve.  Has height = \p height width = \p width, number of channels = \p numChannels, and assumes that stride = \p width.
* @param[in] height  Height of the input image
* @param[in] width  Width of the input image
* @param[in] numChannels  Number of channels in the input image
* @param[in] channelIndex  Index of the channel to convolve
* @param[out] result  Out-of-place result of the image convolved with the kernel.  Is expected that before the call, \p result has size = size of \p image .
*
* @return  true if the convolution succeeded, false otherwise
*/
template <typename kernelT>
bool convolve1DHorizontalPlanar(const kernelT& kernel, const std::vector<float>& image, unsigned int height, unsigned int width, unsigned int numChannels, unsigned int channelIndex, std::vector<float>& result) {
	
	// only operate on odd-sized kernels
	if (kernel.size() % 2 != 1) {
		return false;
	}

	if (channelIndex >= numChannels) {
		return false;
	}

	// size sanity checks
	if (result.size() < image.size()) {
		return false;
	}

	if (height * width * numChannels > image.size()) {
		return false;
	}

	const unsigned int center = static_cast<unsigned int>(kernel.size()) / 2;

	// for a planar image, the selected channel's data is contiguous.  We assume this image has no padding (stride == width, no padding between channels)
	const unsigned int channelStart = height * width * channelIndex;

	// perform a convolution on each row of the input image in the selected channelIndex, storing the result in result
	for (unsigned int row = 0; row < height; row++) {
		const unsigned int rowStart = channelStart + row * width;

		// convolve all pixels in the interior of the image, ignoring the edge pixels
		for (unsigned int col = center; col < width - center; col++) {
			const unsigned int colStart = rowStart + col - center;

			float convolutionResult = 0.0f;
			for (unsigned int kernelIndex = 0; kernelIndex < kernel.size(); kernelIndex++) {
				convolutionResult += kernel[kernelIndex] * image[colStart + kernelIndex];
			}

			result[rowStart + col] = convolutionResult;
		}
	}

	return true;
}

/**
* Performs 1D vertical convolution on a single channel of an input planar format image with the given kernel.  Does not compute convolution around edge.
*
* @tparam kernelT  Kernel type - must have .size() and operator[] (std::array, std::vector, etc)
* @param[in] kernel  1D kernel to convolve with.  Must have odd length.
* @param[in] image  2D multi-channel image to convolve.  Has height = \p height width = \p width, number of channels = \p numChannels, and assumes that stride = \p width.
* @param[in] height  Height of the input image
* @param[in] width  Width of the input image
* @param[in] numChannels  Number of channels in the input image
* @param[in] channelIndex  Index of the channel to convolve
* @param[out] result  Out-of-place result of the image convolved with the kernel.  Is expected that before the call, \p result has size = size of \p image .
*
* @return  true if the convolution succeeded, false otherwise
*/
template <typename kernelT>
bool convolve1DVerticalPlanar(const kernelT& kernel, const std::vector<float>& image, unsigned int height, unsigned int width, unsigned int numChannels, unsigned int channelIndex, std::vector<float>& result) {
	
	// only operate on odd-sized kernels
	if (kernel.size() % 2 != 1) {
		return false;
	}

	if (channelIndex >= numChannels) {
		return false;
	}

	// size sanity checks
	if (result.size() < image.size()) {
		return false;
	}

	if (height * width * numChannels > image.size()) {
		return false;
	}

	const unsigned int center = static_cast<unsigned int>(kernel.size()) / 2;

	// for a planar image, the selected channel's data is contiguous.  We assume this image has no padding (stride == width, no padding between channels)
	const unsigned int channelStart = height * width * channelIndex;

	// perform a convolution on each column of the input image in the selected channelIndex, storing the result in result
	for (unsigned int col = 0; col < width; col++) {
		const unsigned int colStart = channelStart + col;

		// convolve all pixels in the interior of the image, ignoring the edge pixels
		for (unsigned int row = center; row < height - center; row++) {
			const unsigned int rowStart = colStart + (row - center) * width;

			float convolutionResult = 0.0f;
			for (unsigned int kernelIndex = 0; kernelIndex < kernel.size(); kernelIndex++) {
				convolutionResult += kernel[kernelIndex] * image[rowStart + kernelIndex * width];
			}

			result[colStart + row * width] = convolutionResult;
		}
	}

	return true;
}

/**
* Performs 1D horizontal convolution on a single channel of an input interleaved format image with the given kernel.  Does not compute convolution around edge.
*
* @tparam kernelT  Kernel type - must have .size() and operator[] (std::array, std::vector, etc)
* @param[in] kernel  1D kernel to convolve with.  Must have odd length.
* @param[in] image  2D multi-channel image to convolve.  Has height = \p height width = \p width, number of channels = \p numChannels, and assumes that stride = \p width * \p numChannels.
* @param[in] height  Height of the input image
* @param[in] width  Width of the input image
* @param[in] numChannels  Number of channels in the input image
* @param[in] channelIndex  Index of the channel to convolve
* @param[out] result  Out-of-place result of the image convolved with the kernel.  Is expected that before the call, \p result has size = size of \p image .
*
* @return  true if the convolution succeeded, false otherwise
*/
template <typename kernelT>
bool convolve1DHorizontalInterleaved(const kernelT& kernel, const std::vector<float>& image, unsigned int height, unsigned int width, unsigned int numChannels, unsigned int channelIndex, std::vector<float>& result) {
	
	// only operate on odd-sized kernels
	if (kernel.size() % 2 != 1) {
		return false;
	}

	if (channelIndex >= numChannels) {
		return false;
	}

	// size sanity checks
	if (result.size() < image.size()) {
		return false;
	}

	if (height * width * numChannels > image.size()) {
		return false;
	}

	const unsigned int center = static_cast<unsigned int>(kernel.size()) / 2;

	const unsigned int pxStride = numChannels;
	const unsigned int rowStride = pxStride * width;

	// perform a convolution on each row of the input image in the selected channelIndex, storing the result in result
	for (unsigned int row = 0; row < height; row++) {
		const unsigned int rowStart = row * rowStride + channelIndex;

		// convolve all pixels in the interior of the image, ignoring the edge pixels
		for (unsigned int pxCol = center; pxCol < width - center; pxCol++) {
			const unsigned int colStart = rowStart + (pxCol - center) * pxStride;

			float convolutionResult = 0.0f;
			for (unsigned int kernelIndex = 0; kernelIndex < kernel.size(); kernelIndex++) {
				convolutionResult += kernel[kernelIndex] * image[colStart + kernelIndex * pxStride];
			}

			result[rowStart + pxCol * pxStride] = convolutionResult;
		}
	}

	return true;
}

/**
* Performs 1D vertical convolution on a single channel of an input interleaved format image with the given kernel.  Does not compute convolution around edge.
*
* @tparam kernelT  Kernel type - must have .size() and operator[] (std::array, std::vector, etc)
* @param[in] kernel  1D kernel to convolve with.  Must have odd length.
* @param[in] image  2D multi-channel image to convolve.  Has height = \p height width = \p width, number of channels = \p numChannels, and assumes that stride = \p width * \p numChannels.
* @param[in] height  Height of the input image
* @param[in] width  Width of the input image
* @param[in] numChannels  Number of channels in the input image
* @param[in] channelIndex  Index of the channel to convolve
* @param[out] result  Out-of-place result of the image convolved with the kernel.  Is expected that before the call, \p result has size = size of \p image .
*
* @return  true if the convolution succeeded, false otherwise
*/
template <typename kernelT>
bool convolve1DVerticalInterleaved(const kernelT& kernel, const std::vector<float>& image, unsigned int height, unsigned int width, unsigned int numChannels, unsigned int channelIndex, std::vector<float>& result) {
	
	// only operate on odd-sized kernels
	if (kernel.size() % 2 != 1) {
		return false;
	}

	if (channelIndex >= numChannels) {
		return false;
	}

	// size sanity checks
	if (result.size() < image.size()) {
		return false;
	}

	if (height * width * numChannels > image.size()) {
		return false;
	}

	const unsigned int center = static_cast<unsigned int>(kernel.size()) / 2;

	const unsigned int pxStride = numChannels;
	const unsigned int rowStride = pxStride * width;

	// perform a convolution on each column of the input image in the selected channelIndex, storing the result in result
	for (unsigned int col = 0; col < width; col++) {
		const unsigned int colStart = col * pxStride + channelIndex;

		// convolve all pixels in the interior of the image, ignoring the edge pixels
		for (unsigned int row = center; row < height - center; row++) {
			const unsigned int rowStart = colStart + (row - center) * rowStride;

			float convolutionResult = 0.0f;
			for (unsigned int kernelIndex = 0; kernelIndex < kernel.size(); kernelIndex++) {
				convolutionResult += kernel[kernelIndex] * image[rowStart + kernelIndex * rowStride];
			}

			result[colStart + row * rowStride] = convolutionResult;
		}
	}

	return true;
}

/**
 * Transposes the given planar source (src) image
 * 
 * @param[in] src  2D multi-channel planar image to convolve.  Has height = \p height width = \p width, number of channels = \p numChannels, and assumes that stride = \p width.
 * @param[in] height  Height of \p src
 * @param[in] width  Width of \p src
 * @param[in] numChannels  Number of channels in \p src
 * @param[out] dst  The transposed planar image, with height = \p width and width = \p height.  This is expected to have size >= height * width * numChannels.
 * 
 * @return true if the image was successfully transposed, false otherwise.
 */
bool transposePlanar(const std::vector<float>& src, unsigned int height, unsigned int width, unsigned int numChannels, std::vector<float>& dst);