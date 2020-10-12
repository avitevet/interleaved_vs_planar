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


const std::vector<float> planar3channel {
	3.53124f, 7.45078f, 5.21039f, 2.24493f, 4.68696f,
	0.52084f, 2.33007f, 1.00961f, 7.96382f, 3.14524f,
	0.12663f, 6.28619f, 5.25543f, 6.25068f, 6.42683f,
	4.16718f, 5.85775f, 2.83068f, 7.05596f, 7.26622f,
	1.46945f, 0.21148f, 8.41618f, 6.55698f, 7.17606f,

	7.73773f, 7.98205f, 5.70364f, 0.15292f, 7.03645f,
	3.17749f, 5.22830f, 7.26981f, 5.41431f, 0.67898f,
	3.02298f, 6.08901f, 1.75410f, 0.39297f, 2.65367f,
	3.21239f, 7.60296f, 1.41939f, 6.44015f, 1.96547f,
	3.91343f, 1.26121f, 6.67843f, 3.65349f, 5.89449f,

	1.50496f, 1.01108f, 5.87705f, 6.08955f, 0.19340f,
	1.19683f, 4.86358f, 6.37446f, 7.04160f, 1.52744f,
	2.84410f, 6.08736f, 0.41149f, 1.32396f, 8.92492f,
	1.38822f, 4.25869f, 4.58530f, 2.09468f, 5.41935f,
	5.36509f, 4.98096f, 3.59122f, 0.36025f, 3.28838f,
};

const unsigned int planar3channelHeight = 5U;
const unsigned int planar3channelWidth = 5U;
const unsigned int planar3channelChannels = 3U;


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


TEST(planar, horizontal) {
	std::vector<float> dst(planar3channel.size());
	std::fill(dst.begin(), dst.end(), 0.0f);

	ASSERT_TRUE(convolve1DHorizontalPlanar(blur1D, planar3channel, planar3channelHeight, planar3channelWidth, planar3channelChannels, 1, dst));

	const std::vector<float> expectedDst{
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 7.14114f, 4.61287f, 4.29767f, 0,
		0, 5.2252f, 5.97081f, 4.45437f, 0,
		0, 3.62203f, 2.74536f, 1.60025f, 0,
		0, 4.07825f, 5.15417f, 3.275f, 0,
		0, 3.95102f, 3.86438f, 5.4088f, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
	};

	for (auto i = 0U; i < expectedDst.size(); i++) {
		ASSERT_NEAR(expectedDst[i], dst[i], 0.00001f) << "Mismatch at position i = " << i;
	}
}

TEST(planar, vertical) {
	std::vector<float> dst(planar3channel.size());
	std::fill(dst.begin(), dst.end(), 0.0f);

	ASSERT_TRUE(convolve1DVerticalPlanar(blur1D, planar3channel, planar3channelHeight, planar3channelWidth, planar3channelChannels, 0, dst));

	const std::vector<float> expectedDst{
		0, 0, 0, 0, 0,
		1.3929f, 5.35568f, 3.82514f, 5.48648f, 4.75301f,
		1.60488f, 4.82467f, 3.03191f, 7.09015f, 5.61276f,
		1.92109f, 4.11847f, 5.50076f, 6.62121f, 6.95637f,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
	};

	for (auto i = 0U; i < expectedDst.size(); i++) {
		ASSERT_NEAR(expectedDst[i], dst[i], 0.00001f) << "Mismatch at position i = " << i;
	}
}

TEST(planar, transpose) {
	std::vector<float> dst(planar3channel.size());
	std::fill(dst.begin(), dst.end(), 0.0f);

	ASSERT_TRUE(transposePlanar(planar3channel, planar3channelHeight, planar3channelWidth, planar3channelChannels, dst));

	const std::vector<float> expectedDst {
		3.53124f, 0.52084f, 0.12663f, 4.16718f, 1.46945f,
		7.45078f, 2.33007f, 6.28619f, 5.85775f, 0.21148f,
		5.21039f, 1.00961f, 5.25543f, 2.83068f, 8.41618f,
		2.24493f, 7.96382f, 6.25068f, 7.05596f, 6.55698f,
		4.68696f, 3.14524f, 6.42683f, 7.26622f, 7.17606f,

		7.73773f, 3.17749f, 3.02298f, 3.21239f, 3.91343f,
		7.98205f, 5.2283f, 6.08901f, 7.60296f, 1.26121f,
		5.70364f, 7.26981f, 1.7541f, 1.41939f, 6.67843f,
		0.15292f, 5.41431f, 0.39297f, 6.44015f, 3.65349f,
		7.03645f, 0.67898f, 2.65367f, 1.96547f, 5.89449f,

		1.50496f, 1.19683f, 2.8441f, 1.38822f, 5.36509f,
		1.01108f, 4.86358f, 6.08736f, 4.25869f, 4.98096f,
		5.87705f, 6.37446f, 0.41149f, 4.5853f, 3.59122f,
		6.08955f, 7.0416f, 1.32396f, 2.09468f, 0.36025f,
		0.1934f, 1.52744f, 8.92492f, 5.41935f, 3.28838f,
	};

	ASSERT_TRUE(std::equal(expectedDst.begin(), expectedDst.end(), dst.begin()));
}