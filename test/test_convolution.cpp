#include "convolution.h"

#include "gtest/gtest.h"

#include <vector>
#include <array>

const std::vector<float> interleaved2channel {
	1.0f, 0, 2.0f, 0, 3.0f, 0, 1.0f, 0,
	2.0f, 0, 6.0f, 0, 7.0f, 0, 2.0f, 0,
	3.5f, 0, 2.5f, 0, 3.5f, 0, 3.5f, 0,
	4.5f, 0, 6.5f, 0, 7.5f, 0, 4.5f, 0,
};

const unsigned int interleaved2channelHeight = 4U;
const unsigned int interleaved2channelWidth = 4U;
const unsigned int interleaved2channelChannels = 2U;


const std::array<float, 3> blur1D{ {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f } };

TEST(interleaved, horizontal) {
	std::vector<float> dst(interleaved2channel.size());
	std::fill(dst.begin(), dst.end(), 0.0f);

	ASSERT_TRUE(convolve1DHorizontalInterleaved(blur1D, interleaved2channel, interleaved2channelHeight, interleaved2channelWidth, interleaved2channelChannels, 0, dst));

	const std::vector<float> expectedDst{
		0.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 5.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 3.16666675f, 0.0f, 3.16666675f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 6.16666698f, 0.0f, 6.16666698f, 0.0f, 0.0f, 0.0f,
	};

	ASSERT_TRUE(std::equal(expectedDst.begin(), expectedDst.end(), dst.begin()));
}

TEST(interleaved, vertical) {
	std::vector<float> dst(interleaved2channel.size());
	std::fill(dst.begin(), dst.end(), 0.0f);

	ASSERT_TRUE(convolve1DVerticalInterleaved(blur1D, interleaved2channel, interleaved2channelHeight, interleaved2channelWidth, interleaved2channelChannels, 0, dst));

	const std::vector<float> expectedDst{
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		2.16666675f, 0.0f, 3.5f, 0.0f, 4.5f, 0.0f, 2.16666675f, 0.0f,
		3.33333349f, 0.0f, 5.0f, 0.0f, 6.0f, 0.0f, 3.33333349f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	};

	ASSERT_TRUE(std::equal(expectedDst.begin(), expectedDst.end(), dst.begin()));
}