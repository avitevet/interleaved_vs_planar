#include "convolution.h"

#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>
#include <functional>

typedef struct runtimeInfo {
	runtimeInfo(double h, double t, double v)
		: horizontal(h), transpose(t), vertical(v)
	{}

	runtimeInfo()
		: runtimeInfo(0, 0, 0)
	{}

	double GetTotal() const {
		return horizontal + transpose + vertical;
	};

	static runtimeInfo Max() {
		return runtimeInfo{ std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };
	}

	std::string toCsv() const {
		std::stringstream ss;
		ss << horizontal << "," << transpose << "," << vertical << "," << GetTotal();
		return ss.str();
	}

	double horizontal;
	double transpose;
	double vertical;
} tRuntimeInfo;

/**
 * Fills the vector with random values in [0, 1].  This range for float values is typical in
 * image processing, where the value represents 0% to 100% ink coverage of a dot (for print) or light intensity (for screen).
 *
 * @param[out] src  Vector to fill with random values
 */
void fillRandom(std::vector<float>& src) {
	std::default_random_engine generator;
	std::uniform_real_distribution<float> dist(0, 1);

	std::transform(src.begin(), src.end(), src.begin(), [&](float val) {
		return dist(generator);
	});
}

// typedefs for the functions being passed around
template <typename BlurT>
using blurFn = std::function<bool(const BlurT&, const std::vector<float>&, unsigned int, unsigned int, unsigned int, unsigned int, std::vector<float>&)>;
using blur3Fn = blurFn<std::array<float, 3>>;
using blur7Fn = blurFn<std::array<float, 7>>;

using transposeFn = std::function<bool(const std::vector<float>&, unsigned int, unsigned int, unsigned int, std::vector<float>&)>;

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
	blurFn<BlurKernelT> horizontalConvolveFn,
	transposeFn dataTransposeFn,
	blurFn<BlurKernelT> verticalConvolveFn,
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

	// working buffer
	std::vector<float> workingBuffer(dst.size());

	// horizontal convolution in every channel
	const auto horizStart = std::chrono::high_resolution_clock::now();
	for (auto i = 0U; i < depth; i++) {
		horizontalConvolveFn(blurKernel, src, height, width, depth, i, dst);
	}
	const auto horizEnd = std::chrono::high_resolution_clock::now();

	runtimeInfo.horizontal = std::chrono::duration<double>(horizEnd - horizStart).count();

	if (dataTransposeFn) {
		// 1. transpose
		// 2. horizontal convolve
		// 3. transpose again so the results are comparable to a simple horiz/vert convolve

		// Transpose time includes both transposes

		std::vector<float> transposed(dst.size());

		const auto transposeStart = std::chrono::high_resolution_clock::now();
		dataTransposeFn(dst, height, width, depth, transposed);
		const auto transposeEnd = std::chrono::high_resolution_clock::now();

		runtimeInfo.transpose = std::chrono::duration<double>(transposeEnd - transposeStart).count();

		const auto vertStart = std::chrono::high_resolution_clock::now();
		for (auto i = 0U; i < depth; i++) {
			horizontalConvolveFn(blurKernel, transposed, height, width, depth, i, workingBuffer);
		}
		const auto vertEnd = std::chrono::high_resolution_clock::now();

		runtimeInfo.vertical = std::chrono::duration<double>(vertEnd - vertStart).count();

		const auto transpose2Start = std::chrono::high_resolution_clock::now();
		dataTransposeFn(workingBuffer, height, width, depth, dst);
		const auto transpose2End = std::chrono::high_resolution_clock::now();

		runtimeInfo.transpose += std::chrono::duration<double>(transpose2End - transpose2Start).count();

	}
	else {
		// vertical convolution only.  Since this is an out of place operation, the output of the vertical convolve
		// goes into a working buffer, then we copy all the data back into the dst.  We don't include the allocation time
		// in the vertical convolution time, but we do include the final copy.

		const auto vertStart = std::chrono::high_resolution_clock::now();
		for (auto i = 0U; i < depth; i++) {
			verticalConvolveFn(blurKernel, dst, height, width, depth, i, workingBuffer);
		}
		std::copy(workingBuffer.begin(), workingBuffer.end(), dst.begin());
		const auto vertEnd = std::chrono::high_resolution_clock::now();

		runtimeInfo.vertical = std::chrono::duration<double>(vertEnd - vertStart).count();
	}

	return runtimeInfo;
}

/**
 * Converts an image in interleaved layout to planar layout
 *
 * @param[in] interleavedSrc  Input image, in interleaved layout.  Stride is assumed to be \p width * \p depth
 * @param[in] height  Number of rows in the image
 * @param[in] width  Number of columns in the image
 * @param[in] depth  Number of channels in the image
 * @param[out] planarDst  Output image, in planar layout.  Stride is assumed to be \p width
 */
void interleaved2Planar(const std::vector<float>& interleavedSrc, const unsigned int height, const unsigned int width, const unsigned int depth, std::vector<float>& planarDst) {
	for (auto row = 0U; row < height; row++) {
		for (auto col = 0U; col < width; col++) {
			for (auto ch = 0U; ch < depth; ch++) {
				const auto interleavedPosition = (row * width + col) * depth + ch;
				const auto planarPosition = (ch * row * width) + row * width + col;

				planarDst[planarPosition] = interleavedSrc[interleavedPosition];
			}
		}
	}
}

int main(int argc, char ** argv) {
	if (argc != 5) {
		std::cout << "Usage: " << argv[0] << " H W D I" << std::endl;
		std::cout << "H: height of the source matrix to convolve" << std::endl;
		std::cout << "W: width of the source matrix to convolve" << std::endl;
		std::cout << "D: depth (number of channels) of the source matrix to convolve" << std::endl;
		std::cout << "I: Number of iterations to perform.  The minimum total time for a single iteration is reported" << std::endl;
		return 1;
	}

	std::stringstream ssH(argv[1]);
	std::stringstream ssW(argv[2]);
	std::stringstream ssD(argv[3]);
	std::stringstream ssI(argv[4]);

	unsigned int H = 0;
	unsigned int W = 0;
	unsigned int D = 0;
	unsigned int I = 0;

	ssH >> H;
	ssW >> W;
	ssD >> D;
	ssI >> I;

	if ((H == 0) || (W == 0) || (D == 0) || (I == 0)) {
		std::cout << "H, W, D, and I must all be positive integers." << std::endl;
		return 1;
	}

	// create H x W x D float buffers for the source & dst
	const auto numElements = H * W * D;
	std::vector<float> interleavedSrc(numElements);
	std::vector<float> planarSrc(numElements);
	std::vector<float> dst(numElements);

	// fill src with random float values in [0, 1]
	fillRandom(interleavedSrc);
	interleaved2Planar(interleavedSrc, H, W, D, planarSrc);

	blur3Fn horizInterleavedBlur3 = convolve1DHorizontalInterleaved<std::array<float, 3>>;
	blur3Fn vertInterleavedBlur3 = convolve1DVerticalInterleaved<std::array<float, 3>>;
	blur3Fn horizPlanarBlur3 = convolve1DHorizontalPlanar<std::array<float, 3>>;
	blur3Fn vertPlanarBlur3 = convolve1DVerticalPlanar<std::array<float, 3>>;

	blur7Fn horizInterleavedBlur7 = convolve1DHorizontalInterleaved<std::array<float, 7>>;
	blur7Fn vertInterleavedBlur7 = convolve1DVerticalInterleaved<std::array<float, 7>>;
	blur7Fn horizPlanarBlur7 = convolve1DHorizontalPlanar<std::array<float, 7>>;
	blur7Fn vertPlanarBlur7 = convolve1DVerticalPlanar<std::array<float, 7>>;

	transposeFn noTransposeFn;

	tRuntimeInfo minInterleavedBlur3Runtime = tRuntimeInfo::Max();
	tRuntimeInfo minPlanarBlur3Runtime = tRuntimeInfo::Max();
	tRuntimeInfo minInterleavedBlur7Runtime = tRuntimeInfo::Max();
	tRuntimeInfo minPlanarBlur7Runtime = tRuntimeInfo::Max();
	tRuntimeInfo minPlanarBlur7WithTransposeRuntime = tRuntimeInfo::Max();

	for (auto i = 0U; i < I; i++) {
		tRuntimeInfo runtime = measureRuntimeBlur1D<std::array<float, 3>>(interleavedSrc, H, W, D, horizInterleavedBlur3, noTransposeFn, vertInterleavedBlur3, dst);
		if (runtime.GetTotal() < minInterleavedBlur3Runtime.GetTotal()) {
			minInterleavedBlur3Runtime = runtime;
		}
	}

	for (auto i = 0U; i < I; i++) {
		const tRuntimeInfo runtime = measureRuntimeBlur1D<std::array<float, 3>>(planarSrc, H, W, D, horizPlanarBlur3, noTransposeFn, vertPlanarBlur3, dst);
		if (runtime.GetTotal() < minPlanarBlur3Runtime.GetTotal()) {
			minPlanarBlur3Runtime = runtime;
		}
	}

	for (auto i = 0U; i < I; i++) {
		tRuntimeInfo runtime = measureRuntimeBlur1D<std::array<float, 7>>(interleavedSrc, H, W, D, horizInterleavedBlur7, noTransposeFn, vertInterleavedBlur7, dst);
		if (runtime.GetTotal() < minInterleavedBlur7Runtime.GetTotal()) {
			minInterleavedBlur7Runtime = runtime;
		}
	}

	for (auto i = 0U; i < I; i++) {
		tRuntimeInfo runtime = measureRuntimeBlur1D<std::array<float, 7>>(planarSrc, H, W, D, horizPlanarBlur7, noTransposeFn, vertPlanarBlur7, dst);
		if (runtime.GetTotal() < minPlanarBlur7Runtime.GetTotal()) {
			minPlanarBlur7Runtime = runtime;
		}
	}

	for (auto i = 0U; i < I; i++) {
		tRuntimeInfo runtime = measureRuntimeBlur1D<std::array<float, 7>>(planarSrc, H, W, D, horizPlanarBlur7, transposePlanar, vertPlanarBlur7, dst);
		if (runtime.GetTotal() < minPlanarBlur7WithTransposeRuntime.GetTotal()) {
			minPlanarBlur7WithTransposeRuntime = runtime;
		}
	}

	std::cout << "test,horizontal,transpose,vertical,total" << std::endl;
	std::cout << "interleaved3," << minInterleavedBlur3Runtime.toCsv() << std::endl;
	std::cout << "planar3," << minPlanarBlur3Runtime.toCsv() << std::endl;
	std::cout << "interleaved7," << minInterleavedBlur7Runtime.toCsv() << std::endl;
	std::cout << "planar7," << minPlanarBlur7Runtime.toCsv() << std::endl;
	std::cout << "planar7withTranspose," << minPlanarBlur7WithTransposeRuntime.toCsv() << std::endl;

	return 0;
}