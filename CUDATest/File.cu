#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "KernelApi.h"

using namespace std;

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

__global__ void blendImage(const unsigned char* src1,
	const unsigned char* src2, unsigned char* dst, size_t size)
{
	const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < size) {
		// uint8 values are promoted to int before addition, so the sum cannot overflow.
		dst[index] = static_cast<unsigned char>((src1[index] + src2[index]) / 2);
	}
}

bool Proc(vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	//1. Check Input Image
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous())
		return false;
	for (int i = 1; i < Images.size(); i++)
	{
		if (Images[i].empty() || Images[i].cols != Images[0].cols ||
			Images[i].rows != Images[0].rows || Images[i].type() != CV_8UC1 ||
			!Images[i].isContinuous())
			return false;
	}

	const size_t imageSize = Images[0].total();
	const int ImageNum = Images.size();
	constexpr int threadsPerBlock = 256;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	constexpr int totalRuns = 10;
	constexpr int warmupRuns = 5;
	double runTimesMs[totalRuns] = {};

	for (int run = 0; run < totalRuns; ++run)
	{
		const auto start = std::chrono::steady_clock::now();

		//2. Allocate device buffers, transfer data, launch the kernel, and release buffers.
		unsigned char** devPtr_in = new unsigned char*[ImageNum];
		for (int i = 0; i < ImageNum; i++)
		{
			CHECK(cudaMalloc(&devPtr_in[i], imageSize));
			CHECK(cudaMemcpy(devPtr_in[i], Images[i].ptr(), imageSize, cudaMemcpyHostToDevice));
		}
		cv::Mat result(Images[0].rows, Images[0].cols, CV_8UC1);

		int RunIndex = 1;
		while (RunIndex <= 8)
		{
			int Start = 0;
			while (Start + RunIndex < ImageNum)
			{
				blendImage<<<blocksPerGrid,threadsPerBlock>>>(devPtr_in[Start], devPtr_in[Start + RunIndex], devPtr_in[Start], imageSize);
				CHECK(cudaGetLastError());
				Start += (RunIndex * 2);
			}
			RunIndex *= 2;
		}

		CHECK(cudaMemcpy(result.ptr(), devPtr_in[0], imageSize, cudaMemcpyDeviceToHost));

		for (int i = 0; i < ImageNum; i++)
		{
			CHECK(cudaFree(devPtr_in[i]));
		}
		delete[] devPtr_in;

		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("GPU pipeline run %d: %.3f ms%s\n", run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");

		if (run == totalRuns - 1)
			BlendingImage = result;
	}

	double measuredTimeSumMs = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		measuredTimeSumMs += runTimesMs[run];
	printf("GPU pipeline average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, measuredTimeSumMs / (totalRuns - warmupRuns));

	return true;
}
