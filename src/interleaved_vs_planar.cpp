#include "convolution.h"

#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>
#include <functional>

typedef struct runtimeInfo {
	runtimeInfo()
		: horizontal(0), transpose(0), vertical(0)
	{}
	double horizontal;
	double transpose;
	double vertical;

	double GetTotal() const {
		return horizontal + transpose + vertical;
	};
} tRuntimeInfo;

std::string toCsv(const tRuntimeInfo& info) {
	std::stringstream ss;
	ss << info.horizontal << "," << info.transpose << "," << info.vertical << "," << info.GetTotal();
	return ss.str();
}

/**
 * Measures the runtime of convolving a blur kernel of size BlurSpread across all image channels.
 * This only measures runtime, and doesn't check correctness of the convolution routines.  The correctness
 * is assumed and tested in the included tests.
 *
 * @tparam BlurKernel  The array-ish blur kernel.  Required to be odd size
 * @param[in] src  Input data of size height * width * depth
 * @param[in] height  Number of elements in \p src in the height dimension
 * @param[in] width  Number of elements in \p src in the width dimension
 * @param[in] depth  Number of elements in \p src in the depth dimension
 * @param[in] horizontalConvolveFn  The function that performs the horizontal convolution in 1 channel across the whole image
 * @param[in] transposeFn  The optional function that transposes the data.  If this is not empty, the convolution will be performed using the horizontal function, a transpose, and the horizontal function again
 * @param[in] verticalConvolveFn  The function that performs the vertical convolution in 1 channel across the whole image
 * @param[out] dst  Output buffer of size height * width * depth
 */
template <typename BlurKernelT>
tRuntimeInfo measureRuntimeBlur1D(const std::vector<float>& src, const unsigned int height, const unsigned int width, const unsigned int depth,
	std::function<bool(const BlurKernelT&, const std::vector<float>&, unsigned int, unsigned int, unsigned int, unsigned int, std::vector<float>&)> horizontalConvolveFn,
	std::function<bool(const std::vector<float>&, unsigned int, unsigned int, unsigned int, std::vector<float>&)> transposeFn,
	std::function<bool(const BlurKernelT&, const std::vector<float>&, unsigned int, unsigned int, unsigned int, unsigned int, std::vector<float>&)> verticalConvolveFn,
	std::vector<float>& dst) {

	tRuntimeInfo runtimeInfo;

	BlurKernelT blurKernel;
	if (blurKernel.size() % 2 != 1) {
		return runtimeInfo;
	}

	// fill the blur kernel with (1 / size) to get equal contributions from every component
	const auto contribution = 1.0f / static_cast<float>(blurKernel.size());
	std::fill(blurKernel.begin(), blurKernel.end(), contribution);

	// initialize dst with 0s
	std::fill(dst.begin(), dst.end(), 0.0f);

	// horizontal convolution in every channel
	const auto horizStart = std::chrono::high_resolution_clock::now();
	for (auto i = 0U; i < depth; i++) {
		horizontalConvolveFn(blurKernel, src, height, width, depth, i, dst);
	}
	const auto horizEnd = std::chrono::high_resolution_clock::now();

	runtimeInfo.horizontal = std::chrono::duration<double>(horizEnd - horizStart).count();

	if (transposeFn) {
		// out-of-place transpose, then call horizontal convolution again, where the source is the transposed data

		std::vector<float> transposed(dst.size());

		const auto transposeStart = std::chrono::high_resolution_clock::now();
		transposeFn(dst, height, width, depth, transposed);
		const auto transposeEnd = std::chrono::high_resolution_clock::now();

		runtimeInfo.transpose = std::chrono::duration<double>(transposeEnd - transposeStart).count();

		const auto vertStart = std::chrono::high_resolution_clock::now();
		for (auto i = 0U; i < depth; i++) {
			horizontalConvolveFn(blurKernel, transposed, height, width, depth, i, dst);
		}
		const auto vertEnd = std::chrono::high_resolution_clock::now();

		runtimeInfo.vertical = std::chrono::duration<double>(vertEnd - vertStart).count();
	}
	else {
		// vertical convolution only.  Since this is an out of place operation, the output of the vertical convolve
		// is a temp buffer, then we copy all the data back into the dst.  We don't include the allocation time
		// in the vertical convolution time, but we do include the final copy.

		std::vector<float> temp(dst.size());

		const auto vertStart = std::chrono::high_resolution_clock::now();
		for (auto i = 0U; i < depth; i++) {
			verticalConvolveFn(blurKernel, dst, height, width, depth, i, temp);
		}
		std::copy(temp.begin(), temp.end(), dst.begin());
		const auto vertEnd = std::chrono::high_resolution_clock::now();

		runtimeInfo.vertical = std::chrono::duration<double>(vertEnd - vertStart).count();
	}

	return runtimeInfo;
}

int main(int argc, char ** argv) {
	if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " H W D" << std::endl;
		std::cout << "H is the height, W is the width, and D is the depth (number of channels) of the source matrix to convolve." << std::endl;
		return 1;
	}

	std::stringstream ssH(argv[1]);
	std::stringstream ssW(argv[2]);
	std::stringstream ssD(argv[3]);

	unsigned int H = 0;
	unsigned int W = 0;
	unsigned int D = 0;

	ssH >> H;
	ssW >> W;
	ssD >> D;

	if ((H == 0) || (W == 0) || (D == 0)) {
		std::cout << "H, W, and D must all be positive integers." << std::endl;
		return 1;
	}

	// create a H x W x D float matrix for the source, using random float values in [0, 1].  This range for float values is typical in
	// image processing, where the value represents 0% to 100% ink coverage of a dot (for print) or light intensity (for screen).
	std::default_random_engine generator;
	std::uniform_real_distribution<float> dist(0, 1);

	const auto numElements = H * W * D;
	std::vector<float> src(numElements);
	for (auto i = 0U; i < numElements; i++) {
		src[i] = dist(generator);
	}

	std::vector<float> dst(src.size());

	auto horizInterleavedBlur3 = [](
		const std::array<float, 3>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DHorizontalInterleaved<std::array<float, 3>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto vertInterleavedBlur3 = [](
		const std::array<float, 3>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DVerticalInterleaved<std::array<float, 3>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto horizInterleavedBlur7 = [](
		const std::array<float, 7>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DHorizontalInterleaved<std::array<float, 7>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto vertInterleavedBlur7 = [](
		const std::array<float, 7>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DVerticalInterleaved<std::array<float, 7>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto horizPlanarBlur3 = [](
		const std::array<float, 3>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DHorizontalPlanar<std::array<float, 3>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto vertPlanarBlur3 = [](
		const std::array<float, 3>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DVerticalPlanar<std::array<float, 3>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto horizPlanarBlur7 = [](
		const std::array<float, 7>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DHorizontalPlanar<std::array<float, 7>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto vertPlanarBlur7 = [](
		const std::array<float, 7>& kernel,
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		unsigned int channelIndex,
		std::vector<float>& dst) {
			return convolve1DVerticalPlanar<std::array<float, 7>>(kernel, src, height, width, depth, channelIndex, dst);
		};

	auto transposePlanarFn = [](
		const std::vector<float>& src,
		unsigned int height, unsigned int width, unsigned int depth,
		std::vector<float>& dst) {
			return transposePlanar(src, height, width, depth, dst);
		};

	std::function<bool(const std::vector<float>&, unsigned int, unsigned int, unsigned int, std::vector<float>&)> noTransposeFn;

	tRuntimeInfo interleavedBlurSize3Runtime = measureRuntimeBlur1D<std::array<float, 3>>(src, H, W, D, horizInterleavedBlur3, noTransposeFn, vertInterleavedBlur3, dst);
	tRuntimeInfo planarBlurSize3Runtime = measureRuntimeBlur1D<std::array<float, 3>>(src, H, W, D, horizPlanarBlur3, noTransposeFn, vertPlanarBlur3, dst);

	tRuntimeInfo interleavedBlurSize7Runtime = measureRuntimeBlur1D<std::array<float, 7>>(src, H, W, D, horizInterleavedBlur7, noTransposeFn, vertInterleavedBlur7, dst);
	tRuntimeInfo planarBlurSize7Runtime = measureRuntimeBlur1D<std::array<float, 7>>(src, H, W, D, horizPlanarBlur7, noTransposeFn, vertPlanarBlur7, dst);

	tRuntimeInfo planarBlurSize7WithTransposeRuntime = measureRuntimeBlur1D<std::array<float, 7>>(src, H, W, D, horizPlanarBlur7, transposePlanarFn, vertPlanarBlur7, dst);

	std::cout << "test,horizontal,transpose,vertical,total" << std::endl;
	std::cout << "interleaved3," << toCsv(interleavedBlurSize3Runtime) << std::endl;
	std::cout << "planar3," << toCsv(planarBlurSize3Runtime) << std::endl;
	std::cout << "interleaved7," << toCsv(interleavedBlurSize7Runtime) << std::endl;
	std::cout << "planar7," << toCsv(planarBlurSize7Runtime) << std::endl;
	std::cout << "planar7withTranspose," << toCsv(planarBlurSize7WithTransposeRuntime) << std::endl;

	return 0;
}