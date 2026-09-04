#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "KernelApi.h"

inline void checkCuda(cudaError_t result, const char* expression,
	const char* file, int line)
{
	if (result != cudaSuccess) {
		fprintf(stderr,
			"CUDA error at %s:%d, %s: %s\n",
			file, line, expression, cudaGetErrorString(result));
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK(call) \
	checkCuda((call), #call, __FILE__, __LINE__)

// 每个线程独占一个像素偏移，在该偏移上完成整棵二叉归约树。
// 每层结果写回左节点；右节点在被读取后成为下一层可复用的显存区域。
__global__ void blendImageIter(unsigned char* images, size_t imageSize, size_t imageCount)
{
	const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index >= imageSize)
		return;

	// imageCount 必须是 2 的幂：1, 2, 4, ..., imageCount / 2。
	for (size_t pairDistance = 1; pairDistance < imageCount; pairDistance <<= 1) {
		const size_t groupSize = pairDistance << 1;
		for (size_t leftImage = 0; leftImage < imageCount; leftImage += groupSize) {
			const size_t leftOffset = leftImage * imageSize + index;
			const size_t rightOffset = (leftImage + pairDistance) * imageSize + index;
			images[leftOffset] = static_cast<unsigned char>(
				(images[leftOffset] + images[rightOffset]) / 2);
		}
	}
}

bool Proc(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous())
		return false;

	const size_t imageCount = Images.size();
	if ((imageCount & (imageCount - 1)) != 0)
		return false;

	for (size_t imageIndex = 1; imageIndex < imageCount; ++imageIndex) {
		if (Images[imageIndex].empty() || Images[imageIndex].cols != Images[0].cols ||
			Images[imageIndex].rows != Images[0].rows ||
			Images[imageIndex].type() != CV_8UC1 || !Images[imageIndex].isContinuous())
			return false;
	}

	const size_t imageSize = Images[0].total();
	constexpr int threadsPerBlock = 256;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	constexpr int totalRuns = 10;
	constexpr int warmupRuns = 5;
	double runTimesMs[totalRuns] = {};

	for (int run = 0; run < totalRuns; ++run)
	{
		const auto start = std::chrono::steady_clock::now();
		unsigned char* deviceImages = nullptr;
		cv::Mat result(Images[0].rows, Images[0].cols, CV_8UC1);

		// 连续布局：[image0][image1]...[imageN-1]；结果始终原地写回各组的左节点。
		CHECK(cudaMalloc(&deviceImages, imageSize * imageCount));
		for (size_t imageIndex = 0; imageIndex < imageCount; ++imageIndex) {
			CHECK(cudaMemcpy(deviceImages + imageIndex * imageSize, Images[imageIndex].ptr(), imageSize,
				cudaMemcpyHostToDevice));
		}

		blendImageIter<<<blocksPerGrid, threadsPerBlock>>>(deviceImages, imageSize, imageCount);
		CHECK(cudaGetLastError());
		CHECK(cudaMemcpy(result.ptr(), deviceImages, imageSize, cudaMemcpyDeviceToHost));
		CHECK(cudaFree(deviceImages));

		if (run == totalRuns - 1)
			BlendingImage = result;

		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("GPU single-kernel reduction run %d: %.3f ms%s\n", run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}

	double measuredTimeSumMs = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		measuredTimeSumMs += runTimesMs[run];
	printf("GPU single-kernel reduction average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, measuredTimeSumMs / (totalRuns - warmupRuns));

	return true;
}