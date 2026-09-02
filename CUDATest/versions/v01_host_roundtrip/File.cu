#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include "KernelApi.h"

using namespace std;

inline void checkCuda(cudaError_t result, const char* expression,
	const char* file, int line)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d, %s: %s\n",
			file, line, expression, cudaGetErrorString(result));
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

__global__ void blendImage(const unsigned char* src1,
	const unsigned char* src2, unsigned char* dst, size_t size)
{
	const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < size)
		dst[index] = static_cast<unsigned char>((src1[index] + src2[index]) / 2);
}

bool Proc(vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous())
		return false;
	for (int i = 1; i < Images.size(); i++) {
		if (Images[i].empty() || Images[i].cols != Images[0].cols ||
			Images[i].rows != Images[0].rows || Images[i].type() != CV_8UC1 ||
			!Images[i].isContinuous())
			return false;
	}

	const size_t imageSize = Images[0].total();
	constexpr int threadsPerBlock = 256;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	constexpr int totalRuns = 10;
	constexpr int warmupRuns = 5;
	double runTimesMs[totalRuns] = {};

	for (int run = 0; run < totalRuns; ++run) {
		std::queue<cv::Mat> qImages;
		for (const auto& item : Images)
			qImages.push(item.clone());

		const auto start = std::chrono::steady_clock::now();
		unsigned char* devPtr_in1 = nullptr;
		unsigned char* devPtr_in2 = nullptr;
		unsigned char* devPtr_out = nullptr;
		CHECK(cudaMalloc(&devPtr_in1, imageSize));
		CHECK(cudaMalloc(&devPtr_in2, imageSize));
		CHECK(cudaMalloc(&devPtr_out, imageSize));

		while (qImages.size() > 1) {
			cv::Mat Input1 = qImages.front(); qImages.pop();
			cv::Mat Input2 = qImages.front(); qImages.pop();
			CHECK(cudaMemcpy(devPtr_in1, Input1.ptr(), imageSize, cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(devPtr_in2, Input2.ptr(), imageSize, cudaMemcpyHostToDevice));
			blendImage<<<blocksPerGrid, threadsPerBlock>>>(devPtr_in1, devPtr_in2, devPtr_out, imageSize);
			CHECK(cudaGetLastError());
			CHECK(cudaMemcpy(Input1.ptr(), devPtr_out, imageSize, cudaMemcpyDeviceToHost));
			qImages.push(Input1);
		}

		cv::Mat result = qImages.front().clone();
		CHECK(cudaFree(devPtr_out));
		CHECK(cudaFree(devPtr_in2));
		CHECK(cudaFree(devPtr_in1));
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
