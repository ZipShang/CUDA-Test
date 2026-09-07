#include <stdio.h>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <thread>
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

// 基础的图像融合核函数，支持原地写入
__global__ void blendImage(const unsigned char* src1,
	const unsigned char* src2, unsigned char* dst, size_t size)
{
	const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < size) {
		// uint8 values are promoted to int before addition, so the sum cannot overflow.
		dst[index] = static_cast<unsigned char>((src1[index] + src2[index]) / 2);
	}
}

// 每个线程独占一个像素偏移，在该偏移上完成整棵二叉归约树。
// 每层结果写回左节点；
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

constexpr int kImageCount = 16;

// Proc7 专用：完整扫描每张输入图，计算亮度均值与标准差所需的平方和。
// grid 只使用每个 SM 一个 block，使该质量检测可与全分辨率融合核函数并发驻留。
__global__ void proc7InputQuality(const unsigned char* deviceImages, size_t imageSize,
	unsigned long long* sums, unsigned long long* squaredSums)
{
	const int threadIndex = threadIdx.x;
	const size_t firstPixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIndex;
	const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
	extern __shared__ unsigned long long sharedValues[];
	unsigned long long* sharedSums = sharedValues;
	unsigned long long* sharedSquaredSums = sharedValues + blockDim.x;

	for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
		const unsigned char* image = deviceImages + static_cast<size_t>(imageIndex) * imageSize;
		unsigned long long localSum = 0;
		unsigned long long localSquaredSum = 0;
		for (size_t pixel = firstPixel; pixel < imageSize; pixel += stride) {
			const unsigned long long value = image[pixel];
			localSum += value;
			localSquaredSum += value * value;
		}

		sharedSums[threadIndex] = localSum;
		sharedSquaredSums[threadIndex] = localSquaredSum;
		__syncthreads();
		for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
			if (threadIndex < offset) {
				sharedSums[threadIndex] += sharedSums[threadIndex + offset];
				sharedSquaredSums[threadIndex] += sharedSquaredSums[threadIndex + offset];
			}
			__syncthreads();
		}
		if (threadIndex == 0) {
			atomicAdd(&sums[imageIndex], sharedSums[0]);
			atomicAdd(&squaredSums[imageIndex], sharedSquaredSums[0]);
		}
		__syncthreads();
	}
}

namespace {

constexpr int kDeviceBufferCount = 5;
constexpr int kReductionLevels = 5;
constexpr int kKernelCount = kImageCount - 1;

struct Node {
	// 表示连续 2^level 张原图归约后的结果，ready 标记其可被计算流读取的时刻。
	int firstImage = 0;
	int level = 0;
	int bufferIndex = 0;
	cudaEvent_t ready = nullptr;
	};

struct ReusableBuffer {
	// buffer 只有在 reusableAfter 对应的 kernel 读完后才能被传输流覆盖。
	int bufferIndex = 0;
	cudaEvent_t reusableAfter = nullptr;
	};

} // namespace

// null stream，每次图像融合实现 H2D->Kernel->D2H,回传中间图像。包含显存复用
bool Proc1(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
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

// null stream，将图像全部拷贝到显存，在显存中实现多轮图像融合，不将中间数据传回host,包含显存复用
bool Proc2(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
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

// Proc2 的线程块大小测试重载：完整执行 16 图设备端归约，并单独记录 15 次融合 kernel 的耗时。
bool Proc2(std::vector<cv::Mat>& images, cv::Mat& blendingImage, int ThreadNumPerBlock)
{
	if (images.empty() || images[0].empty() || images[0].type() != CV_8UC1 ||
		images[0].rows == 0 || images[0].cols == 0 || !images[0].isContinuous() ||
		images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (images[imageIndex].empty() || images[imageIndex].size() != images[0].size() ||
			images[imageIndex].type() != CV_8UC1 || !images[imageIndex].isContinuous())
			return false;

	cudaDeviceProp deviceProperties = {};
	CHECK(cudaGetDeviceProperties(&deviceProperties, 0));
	if (ThreadNumPerBlock <= 0 || ThreadNumPerBlock > deviceProperties.maxThreadsPerBlock) {
		fprintf(stderr, "Invalid Proc2 overload block size %d; device maximum is %d.\n",
			ThreadNumPerBlock, deviceProperties.maxThreadsPerBlock);
		return false;
	}

	const size_t imageSize = images[0].total();
	constexpr int totalRuns = 10, warmupRuns = 5;
	const int blocksPerGrid = static_cast<int>((imageSize + ThreadNumPerBlock - 1) / ThreadNumPerBlock);
	double wallTimesMs[totalRuns] = {};
	float kernelTimesMs[totalRuns] = {};

	for (int run = 0; run < totalRuns; ++run) {
		const auto wallStart = std::chrono::steady_clock::now();
		cv::Mat result(images[0].rows, images[0].cols, CV_8UC1);
		unsigned char* deviceImages = nullptr;
		cudaEvent_t kernelStart = nullptr;
		cudaEvent_t kernelEnd = nullptr;

		CHECK(cudaMalloc(&deviceImages, imageSize * kImageCount));
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
			CHECK(cudaMemcpy(deviceImages + static_cast<size_t>(imageIndex) * imageSize,
				images[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice));
		}
		CHECK(cudaEventCreate(&kernelStart));
		CHECK(cudaEventCreate(&kernelEnd));
		CHECK(cudaEventRecord(kernelStart));

		// 与 Proc2/Proc4 相同的二叉归约树；唯一实验变量是 ThreadNumPerBlock。
		int pairDistance = 1;
		while (pairDistance < kImageCount) {
			for (int firstImage = 0; firstImage + pairDistance < kImageCount;
				firstImage += pairDistance * 2) {
				unsigned char* left = deviceImages + static_cast<size_t>(firstImage) * imageSize;
				unsigned char* right = deviceImages + static_cast<size_t>(firstImage + pairDistance) * imageSize;
				blendImage << <blocksPerGrid, ThreadNumPerBlock >> > (left, right, left, imageSize);
				CHECK(cudaGetLastError());
			}
			pairDistance <<= 1;
		}

		CHECK(cudaEventRecord(kernelEnd));
		CHECK(cudaEventSynchronize(kernelEnd));
		CHECK(cudaEventElapsedTime(&kernelTimesMs[run], kernelStart, kernelEnd));
		CHECK(cudaMemcpy(result.ptr(), deviceImages, imageSize, cudaMemcpyDeviceToHost));
		CHECK(cudaEventDestroy(kernelEnd));
		CHECK(cudaEventDestroy(kernelStart));
		CHECK(cudaFree(deviceImages));

		if (run == totalRuns - 1)
			blendingImage = result;
		wallTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - wallStart).count();
		printf("Proc2 overload block %d run %d: wall = %.3f ms, kernels = %.3f ms%s\n",
			ThreadNumPerBlock, run + 1, wallTimesMs[run], kernelTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}

	double wallTotalMs = 0.0;
	double kernelTotalMs = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run) {
		wallTotalMs += wallTimesMs[run];
		kernelTotalMs += kernelTimesMs[run];
	}
	printf("Proc2 overload block %d average (runs %d-%d): wall = %.3f ms, 15 kernels = %.3f ms\n",
		ThreadNumPerBlock, warmupRuns + 1, totalRuns,
		wallTotalMs / (totalRuns - warmupRuns), kernelTotalMs / (totalRuns - warmupRuns));
	return true;
}

// compute stream & copy stream,最大化显存复用，将图像传输流与核函数调用流最大程度并行
bool Proc3(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous() ||
		Images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (Images[imageIndex].empty() || Images[imageIndex].size() != Images[0].size() ||
			Images[imageIndex].type() != CV_8UC1 || !Images[imageIndex].isContinuous())
			return false;
	const size_t imageSize = Images[0].total();
	constexpr int threadsPerBlock = 256, totalRuns = 10, warmupRuns = 5;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	double runTimesMs[totalRuns] = {};

	auto runPipeline = [&](std::vector<cv::Mat>& pipelineImages, cv::Mat& result,
		bool hostMemoryAlreadyPinned) {
		if (!hostMemoryAlreadyPinned)
			result.create(pipelineImages[0].rows, pipelineImages[0].cols, CV_8UC1);
		std::array<unsigned char*, kDeviceBufferCount> deviceBuffers = {};
		std::array<cudaEvent_t, kImageCount> copyDoneEvents = {};
		std::array<cudaEvent_t, kKernelCount> kernelDoneEvents = {};
		cudaStream_t copyStream = nullptr;
		cudaStream_t computeStream = nullptr;
		if (!hostMemoryAlreadyPinned) {
			for (cv::Mat& image : pipelineImages) CHECK(cudaHostRegister(image.ptr(), imageSize, cudaHostRegisterDefault));
			CHECK(cudaHostRegister(result.ptr(), imageSize, cudaHostRegisterDefault));
		}
		for (unsigned char*& buffer : deviceBuffers) CHECK(cudaMalloc(&buffer, imageSize));
		for (cudaEvent_t& event : copyDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		for (cudaEvent_t& event : kernelDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
		std::array<Node, kReductionLevels> pendingNodes = {};
		std::array<bool, kReductionLevels> hasPendingNode = {};
		std::array<ReusableBuffer, kDeviceBufferCount> reusableBuffers = {};
		int reusableBufferCount = kDeviceBufferCount, copyEventIndex = 0, kernelEventIndex = 0;
		for (int i = 0; i < kDeviceBufferCount; ++i) reusableBuffers[i] = { i, nullptr };
		auto releaseBuffer = [&](int index, cudaEvent_t after) {
			if (reusableBufferCount >= kDeviceBufferCount) { fprintf(stderr, "Buffer pool overflow.\n"); std::exit(EXIT_FAILURE); }
			reusableBuffers[reusableBufferCount++] = { index, after };
		};
		auto acquireBuffer = [&]() {
			if (reusableBufferCount == 0) { fprintf(stderr, "Buffer pool exhausted.\n"); std::exit(EXIT_FAILURE); }
			int slot = 0;
			for (int i = 0; i < reusableBufferCount; ++i) if (reusableBuffers[i].reusableAfter == nullptr) { slot = i; break; }
			const ReusableBuffer buffer = reusableBuffers[slot];
			reusableBuffers[slot] = reusableBuffers[--reusableBufferCount];
			if (buffer.reusableAfter != nullptr) CHECK(cudaStreamWaitEvent(copyStream, buffer.reusableAfter, 0));
			return buffer.bufferIndex;
		};
		auto mergeNodes = [&](const Node& left, const Node& right) {
			if (left.level != right.level || left.firstImage + (1 << left.level) != right.firstImage || kernelEventIndex >= kKernelCount) {
				fprintf(stderr, "Invalid image reduction tree.\n"); std::exit(EXIT_FAILURE);
			}
			CHECK(cudaStreamWaitEvent(computeStream, left.ready, 0));
			CHECK(cudaStreamWaitEvent(computeStream, right.ready, 0));
			blendImage<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(deviceBuffers[left.bufferIndex], deviceBuffers[right.bufferIndex], deviceBuffers[left.bufferIndex], imageSize);
			CHECK(cudaGetLastError());
			cudaEvent_t done = kernelDoneEvents[kernelEventIndex++];
			CHECK(cudaEventRecord(done, computeStream));
			releaseBuffer(right.bufferIndex, done);
			return Node{ left.firstImage, left.level + 1, left.bufferIndex, done };
		};
		auto addNode = [&](Node node) {
			while (hasPendingNode[node.level]) { const Node left = pendingNodes[node.level]; hasPendingNode[node.level] = false; node = mergeNodes(left, node); }
			pendingNodes[node.level] = node; hasPendingNode[node.level] = true;
		};
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
			const int bufferIndex = acquireBuffer();
			cudaEvent_t done = copyDoneEvents[copyEventIndex++];
			CHECK(cudaMemcpyAsync(deviceBuffers[bufferIndex], pipelineImages[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice, copyStream));
			CHECK(cudaEventRecord(done, copyStream));
			addNode(Node{ imageIndex, 0, bufferIndex, done });
		}
		if (kernelEventIndex != kKernelCount || !hasPendingNode[kReductionLevels - 1]) { fprintf(stderr, "Incomplete image reduction tree.\n"); std::exit(EXIT_FAILURE); }
		const Node finalNode = pendingNodes[kReductionLevels - 1];
		CHECK(cudaStreamWaitEvent(copyStream, finalNode.ready, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceBuffers[finalNode.bufferIndex], imageSize, cudaMemcpyDeviceToHost, copyStream));
		CHECK(cudaStreamSynchronize(copyStream));
		for (cudaEvent_t event : kernelDoneEvents) CHECK(cudaEventDestroy(event));
		for (cudaEvent_t event : copyDoneEvents) CHECK(cudaEventDestroy(event));
		CHECK(cudaStreamDestroy(computeStream)); CHECK(cudaStreamDestroy(copyStream));
		for (unsigned char* buffer : deviceBuffers) CHECK(cudaFree(buffer));
		if (!hostMemoryAlreadyPinned) {
			CHECK(cudaHostUnregister(result.ptr()));
			for (cv::Mat& image : pipelineImages) CHECK(cudaHostUnregister(image.ptr()));
		}
	};

	for (int run = 0; run < totalRuns; ++run) {
		const auto start = std::chrono::steady_clock::now();
		cv::Mat result; runPipeline(Images, result, false);
		if (run == totalRuns - 1) BlendingImage = result;
		runTimesMs[run] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		printf("GPU async pipeline run %d: %.3f ms%s\n", run + 1, runTimesMs[run], run < warmupRuns ? " (warm-up)" : "");
	}
	double total = 0.0; for (int run = warmupRuns; run < totalRuns; ++run) total += runTimesMs[run];
	printf("GPU async pipeline average (runs %d-%d): %.3f ms\n", warmupRuns + 1, totalRuns, total / (totalRuns - warmupRuns));
	return true;
}

// 申请连续显存，将图像数据全部传入显存，在一个核函数中实现多层图像融合
bool Proc4(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
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

		blendImageIter << <blocksPerGrid, threadsPerBlock >> > (deviceImages, imageSize, imageCount);
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

// 三个主机线程各自提交一条与 Proc3 相同的双 stream GPU 流水线。
bool Proc5(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous() ||
		Images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (Images[imageIndex].empty() || Images[imageIndex].size() != Images[0].size() ||
			Images[imageIndex].type() != CV_8UC1 || !Images[imageIndex].isContinuous())
			return false;
	const size_t imageSize = Images[0].total();
	constexpr int threadsPerBlock = 256, totalRuns = 10, warmupRuns = 5;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	double runTimesMs[totalRuns] = {};
	std::array<std::vector<cv::Mat>, 3> pipelineInputs;
	for (std::vector<cv::Mat>& input : pipelineInputs) {
		input.reserve(kImageCount);
		for (const cv::Mat& image : Images) input.push_back(image.clone());
	}

	auto runPipeline = [&](std::vector<cv::Mat>& pipelineImages, cv::Mat& result,
		bool hostMemoryAlreadyPinned) {
		if (!hostMemoryAlreadyPinned)
			result.create(pipelineImages[0].rows, pipelineImages[0].cols, CV_8UC1);
		std::array<unsigned char*, kDeviceBufferCount> deviceBuffers = {};
		std::array<cudaEvent_t, kImageCount> copyDoneEvents = {};
		std::array<cudaEvent_t, kKernelCount> kernelDoneEvents = {};
		cudaStream_t copyStream = nullptr;
		cudaStream_t computeStream = nullptr;
		if (!hostMemoryAlreadyPinned) {
			for (cv::Mat& image : pipelineImages) CHECK(cudaHostRegister(image.ptr(), imageSize, cudaHostRegisterDefault));
			CHECK(cudaHostRegister(result.ptr(), imageSize, cudaHostRegisterDefault));
		}
		for (unsigned char*& buffer : deviceBuffers) CHECK(cudaMalloc(&buffer, imageSize));
		for (cudaEvent_t& event : copyDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		for (cudaEvent_t& event : kernelDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
		std::array<Node, kReductionLevels> pendingNodes = {};
		std::array<bool, kReductionLevels> hasPendingNode = {};
		std::array<ReusableBuffer, kDeviceBufferCount> reusableBuffers = {};
		int reusableBufferCount = kDeviceBufferCount, copyEventIndex = 0, kernelEventIndex = 0;
		for (int i = 0; i < kDeviceBufferCount; ++i) reusableBuffers[i] = { i, nullptr };
		auto releaseBuffer = [&](int index, cudaEvent_t after) {
			if (reusableBufferCount >= kDeviceBufferCount) { fprintf(stderr, "Buffer pool overflow.\n"); std::exit(EXIT_FAILURE); }
			reusableBuffers[reusableBufferCount++] = { index, after };
		};
		auto acquireBuffer = [&]() {
			if (reusableBufferCount == 0) { fprintf(stderr, "Buffer pool exhausted.\n"); std::exit(EXIT_FAILURE); }
			int slot = 0;
			for (int i = 0; i < reusableBufferCount; ++i) if (reusableBuffers[i].reusableAfter == nullptr) { slot = i; break; }
			const ReusableBuffer buffer = reusableBuffers[slot];
			reusableBuffers[slot] = reusableBuffers[--reusableBufferCount];
			if (buffer.reusableAfter != nullptr) CHECK(cudaStreamWaitEvent(copyStream, buffer.reusableAfter, 0));
			return buffer.bufferIndex;
		};
		auto mergeNodes = [&](const Node& left, const Node& right) {
			if (left.level != right.level || left.firstImage + (1 << left.level) != right.firstImage || kernelEventIndex >= kKernelCount) {
				fprintf(stderr, "Invalid image reduction tree.\n"); std::exit(EXIT_FAILURE);
			}
			CHECK(cudaStreamWaitEvent(computeStream, left.ready, 0));
			CHECK(cudaStreamWaitEvent(computeStream, right.ready, 0));
			blendImage<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(deviceBuffers[left.bufferIndex], deviceBuffers[right.bufferIndex], deviceBuffers[left.bufferIndex], imageSize);
			CHECK(cudaGetLastError());
			cudaEvent_t done = kernelDoneEvents[kernelEventIndex++];
			CHECK(cudaEventRecord(done, computeStream));
			releaseBuffer(right.bufferIndex, done);
			return Node{ left.firstImage, left.level + 1, left.bufferIndex, done };
		};
		auto addNode = [&](Node node) {
			while (hasPendingNode[node.level]) { const Node left = pendingNodes[node.level]; hasPendingNode[node.level] = false; node = mergeNodes(left, node); }
			pendingNodes[node.level] = node; hasPendingNode[node.level] = true;
		};
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
			const int bufferIndex = acquireBuffer();
			cudaEvent_t done = copyDoneEvents[copyEventIndex++];
			CHECK(cudaMemcpyAsync(deviceBuffers[bufferIndex], pipelineImages[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice, copyStream));
			CHECK(cudaEventRecord(done, copyStream));
			addNode(Node{ imageIndex, 0, bufferIndex, done });
		}
		if (kernelEventIndex != kKernelCount || !hasPendingNode[kReductionLevels - 1]) { fprintf(stderr, "Incomplete image reduction tree.\n"); std::exit(EXIT_FAILURE); }
		const Node finalNode = pendingNodes[kReductionLevels - 1];
		CHECK(cudaStreamWaitEvent(copyStream, finalNode.ready, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceBuffers[finalNode.bufferIndex], imageSize, cudaMemcpyDeviceToHost, copyStream));
		CHECK(cudaStreamSynchronize(copyStream));
		for (cudaEvent_t event : kernelDoneEvents) CHECK(cudaEventDestroy(event));
		for (cudaEvent_t event : copyDoneEvents) CHECK(cudaEventDestroy(event));
		CHECK(cudaStreamDestroy(computeStream)); CHECK(cudaStreamDestroy(copyStream));
		for (unsigned char* buffer : deviceBuffers) CHECK(cudaFree(buffer));
		if (!hostMemoryAlreadyPinned) {
			CHECK(cudaHostUnregister(result.ptr()));
			for (cv::Mat& image : pipelineImages) CHECK(cudaHostUnregister(image.ptr()));
		}
	};

	for (int run = 0; run < totalRuns; ++run) {
		const auto start = std::chrono::steady_clock::now();
		std::array<cv::Mat, 3> results;
		std::array<std::thread, 3> threads;
		for (int i = 0; i < 3; ++i) threads[i] = std::thread([&, i] { runPipeline(pipelineInputs[i], results[i], false); });
		for (std::thread& thread : threads) thread.join();
		if (run == totalRuns - 1) BlendingImage = results[0];
		runTimesMs[run] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		printf("Three host threads + GPU async pipelines run %d: %.3f ms%s\n", run + 1, runTimesMs[run], run < warmupRuns ? " (warm-up)" : "");
	}
	double total = 0.0; for (int run = warmupRuns; run < totalRuns; ++run) total += runTimesMs[run];
	printf("Three host threads + GPU async pipelines average (runs %d-%d): %.3f ms\n", warmupRuns + 1, totalRuns, total / (totalRuns - warmupRuns));
	return true;
}

// 在调用线程串行提交与 Proc5 数量相同的 GPU 流水线。
bool Proc5_Compare(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous() ||
		Images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (Images[imageIndex].empty() || Images[imageIndex].size() != Images[0].size() ||
			Images[imageIndex].type() != CV_8UC1 || !Images[imageIndex].isContinuous())
			return false;
	const size_t imageSize = Images[0].total();
	constexpr int threadsPerBlock = 256, totalRuns = 10, warmupRuns = 5;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	double runTimesMs[totalRuns] = {};
	std::array<std::vector<cv::Mat>, 3> pipelineInputs;
	for (std::vector<cv::Mat>& input : pipelineInputs) {
		input.reserve(kImageCount);
		for (const cv::Mat& image : Images) input.push_back(image.clone());
	}

	auto runPipeline = [&](std::vector<cv::Mat>& pipelineImages, cv::Mat& result,
		bool hostMemoryAlreadyPinned) {
		if (!hostMemoryAlreadyPinned)
			result.create(pipelineImages[0].rows, pipelineImages[0].cols, CV_8UC1);
		std::array<unsigned char*, kDeviceBufferCount> deviceBuffers = {};
		std::array<cudaEvent_t, kImageCount> copyDoneEvents = {};
		std::array<cudaEvent_t, kKernelCount> kernelDoneEvents = {};
		cudaStream_t copyStream = nullptr;
		cudaStream_t computeStream = nullptr;
		if (!hostMemoryAlreadyPinned) {
			for (cv::Mat& image : pipelineImages) CHECK(cudaHostRegister(image.ptr(), imageSize, cudaHostRegisterDefault));
			CHECK(cudaHostRegister(result.ptr(), imageSize, cudaHostRegisterDefault));
		}
		for (unsigned char*& buffer : deviceBuffers) CHECK(cudaMalloc(&buffer, imageSize));
		for (cudaEvent_t& event : copyDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		for (cudaEvent_t& event : kernelDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
		std::array<Node, kReductionLevels> pendingNodes = {};
		std::array<bool, kReductionLevels> hasPendingNode = {};
		std::array<ReusableBuffer, kDeviceBufferCount> reusableBuffers = {};
		int reusableBufferCount = kDeviceBufferCount, copyEventIndex = 0, kernelEventIndex = 0;
		for (int i = 0; i < kDeviceBufferCount; ++i) reusableBuffers[i] = { i, nullptr };
		auto releaseBuffer = [&](int index, cudaEvent_t after) {
			if (reusableBufferCount >= kDeviceBufferCount) { fprintf(stderr, "Buffer pool overflow.\n"); std::exit(EXIT_FAILURE); }
			reusableBuffers[reusableBufferCount++] = { index, after };
		};
		auto acquireBuffer = [&]() {
			if (reusableBufferCount == 0) { fprintf(stderr, "Buffer pool exhausted.\n"); std::exit(EXIT_FAILURE); }
			int slot = 0;
			for (int i = 0; i < reusableBufferCount; ++i) if (reusableBuffers[i].reusableAfter == nullptr) { slot = i; break; }
			const ReusableBuffer buffer = reusableBuffers[slot];
			reusableBuffers[slot] = reusableBuffers[--reusableBufferCount];
			if (buffer.reusableAfter != nullptr) CHECK(cudaStreamWaitEvent(copyStream, buffer.reusableAfter, 0));
			return buffer.bufferIndex;
		};
		auto mergeNodes = [&](const Node& left, const Node& right) {
			if (left.level != right.level || left.firstImage + (1 << left.level) != right.firstImage || kernelEventIndex >= kKernelCount) {
				fprintf(stderr, "Invalid image reduction tree.\n"); std::exit(EXIT_FAILURE);
			}
			CHECK(cudaStreamWaitEvent(computeStream, left.ready, 0));
			CHECK(cudaStreamWaitEvent(computeStream, right.ready, 0));
			blendImage<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(deviceBuffers[left.bufferIndex], deviceBuffers[right.bufferIndex], deviceBuffers[left.bufferIndex], imageSize);
			CHECK(cudaGetLastError());
			cudaEvent_t done = kernelDoneEvents[kernelEventIndex++];
			CHECK(cudaEventRecord(done, computeStream));
			releaseBuffer(right.bufferIndex, done);
			return Node{ left.firstImage, left.level + 1, left.bufferIndex, done };
		};
		auto addNode = [&](Node node) {
			while (hasPendingNode[node.level]) { const Node left = pendingNodes[node.level]; hasPendingNode[node.level] = false; node = mergeNodes(left, node); }
			pendingNodes[node.level] = node; hasPendingNode[node.level] = true;
		};
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
			const int bufferIndex = acquireBuffer();
			cudaEvent_t done = copyDoneEvents[copyEventIndex++];
			CHECK(cudaMemcpyAsync(deviceBuffers[bufferIndex], pipelineImages[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice, copyStream));
			CHECK(cudaEventRecord(done, copyStream));
			addNode(Node{ imageIndex, 0, bufferIndex, done });
		}
		if (kernelEventIndex != kKernelCount || !hasPendingNode[kReductionLevels - 1]) { fprintf(stderr, "Incomplete image reduction tree.\n"); std::exit(EXIT_FAILURE); }
		const Node finalNode = pendingNodes[kReductionLevels - 1];
		CHECK(cudaStreamWaitEvent(copyStream, finalNode.ready, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceBuffers[finalNode.bufferIndex], imageSize, cudaMemcpyDeviceToHost, copyStream));
		CHECK(cudaStreamSynchronize(copyStream));
		for (cudaEvent_t event : kernelDoneEvents) CHECK(cudaEventDestroy(event));
		for (cudaEvent_t event : copyDoneEvents) CHECK(cudaEventDestroy(event));
		CHECK(cudaStreamDestroy(computeStream)); CHECK(cudaStreamDestroy(copyStream));
		for (unsigned char* buffer : deviceBuffers) CHECK(cudaFree(buffer));
		if (!hostMemoryAlreadyPinned) {
			CHECK(cudaHostUnregister(result.ptr()));
			for (cv::Mat& image : pipelineImages) CHECK(cudaHostUnregister(image.ptr()));
		}
	};

	for (int run = 0; run < totalRuns; ++run) {
		const auto start = std::chrono::steady_clock::now();
		std::array<cv::Mat, 3> results;
		for (int i = 0; i < 3; ++i) runPipeline(pipelineInputs[i], results[i], false);
		if (run == totalRuns - 1) BlendingImage = results[0];
		runTimesMs[run] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		printf("Single host thread + GPU async pipelines run %d: %.3f ms%s\n", run + 1, runTimesMs[run], run < warmupRuns ? " (warm-up)" : "");
	}
	double total = 0.0; for (int run = warmupRuns; run < totalRuns; ++run) total += runTimesMs[run];
	printf("Single host thread + GPU async pipelines average (runs %d-%d): %.3f ms\n", warmupRuns + 1, totalRuns, total / (totalRuns - warmupRuns));
	return true;
}

// 申请固定缓存，对固定缓存进行复用，CUDA调用逻辑同Proc3
bool Proc6(std::vector<cv::Mat>& images, cv::Mat& blendingImage)
{
	if (images.empty() || images[0].empty() || images[0].type() != CV_8UC1 ||
		images[0].rows == 0 || images[0].cols == 0 || !images[0].isContinuous() ||
		images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (images[imageIndex].empty() || images[imageIndex].size() != images[0].size() ||
			images[imageIndex].type() != CV_8UC1 || !images[imageIndex].isContinuous())
			return false;

	const size_t imageSize = images[0].total();
	const size_t pinnedImageCount = kImageCount + 1;
	const size_t pinnedStagingSize = imageSize * pinnedImageCount;
	constexpr int threadsPerBlock = 256;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	constexpr int totalRuns = 10;
	constexpr int warmupRuns = 5;
	double runTimesMs[totalRuns] = {};
	double stagingCopyTimesMs[totalRuns] = {};

	auto runPipeline = [&](std::vector<cv::Mat>& pipelineImages, cv::Mat& result)
	{

		std::array<unsigned char*, kDeviceBufferCount> deviceBuffers = {};
		std::array<cudaEvent_t, kImageCount> copyDoneEvents = {};
		std::array<cudaEvent_t, kKernelCount> kernelDoneEvents = {};
		cudaStream_t copyStream = nullptr;
		cudaStream_t computeStream = nullptr;

		for (unsigned char*& buffer : deviceBuffers) CHECK(cudaMalloc(&buffer, imageSize));
		for (cudaEvent_t& event : copyDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		for (cudaEvent_t& event : kernelDoneEvents) CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
		CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
		std::array<Node, kReductionLevels> pendingNodes = {};
		std::array<bool, kReductionLevels> hasPendingNode = {};
		std::array<ReusableBuffer, kDeviceBufferCount> reusableBuffers = {};
		int reusableBufferCount = kDeviceBufferCount, copyEventIndex = 0, kernelEventIndex = 0;
		for (int i = 0; i < kDeviceBufferCount; ++i) reusableBuffers[i] = { i, nullptr };
		auto releaseBuffer = [&](int index, cudaEvent_t after) {
			if (reusableBufferCount >= kDeviceBufferCount) { fprintf(stderr, "Buffer pool overflow.\n"); std::exit(EXIT_FAILURE); }
			reusableBuffers[reusableBufferCount++] = { index, after };
		};
		auto acquireBuffer = [&]() {
			if (reusableBufferCount == 0) { fprintf(stderr, "Buffer pool exhausted.\n"); std::exit(EXIT_FAILURE); }
			int slot = 0;
			for (int i = 0; i < reusableBufferCount; ++i) if (reusableBuffers[i].reusableAfter == nullptr) { slot = i; break; }
			const ReusableBuffer buffer = reusableBuffers[slot];
			reusableBuffers[slot] = reusableBuffers[--reusableBufferCount];
			if (buffer.reusableAfter != nullptr) CHECK(cudaStreamWaitEvent(copyStream, buffer.reusableAfter, 0));
			return buffer.bufferIndex;
		};
		auto mergeNodes = [&](const Node& left, const Node& right) {
			if (left.level != right.level || left.firstImage + (1 << left.level) != right.firstImage || kernelEventIndex >= kKernelCount) {
				fprintf(stderr, "Invalid image reduction tree.\n"); std::exit(EXIT_FAILURE);
			}
			CHECK(cudaStreamWaitEvent(computeStream, left.ready, 0));
			CHECK(cudaStreamWaitEvent(computeStream, right.ready, 0));
			blendImage<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(deviceBuffers[left.bufferIndex], deviceBuffers[right.bufferIndex], deviceBuffers[left.bufferIndex], imageSize);
			CHECK(cudaGetLastError());
			cudaEvent_t done = kernelDoneEvents[kernelEventIndex++];
			CHECK(cudaEventRecord(done, computeStream));
			releaseBuffer(right.bufferIndex, done);
			return Node{ left.firstImage, left.level + 1, left.bufferIndex, done };
		};
		auto addNode = [&](Node node) {
			while (hasPendingNode[node.level]) { const Node left = pendingNodes[node.level]; hasPendingNode[node.level] = false; node = mergeNodes(left, node); }
			pendingNodes[node.level] = node; hasPendingNode[node.level] = true;
		};
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
			const int bufferIndex = acquireBuffer();
			cudaEvent_t done = copyDoneEvents[copyEventIndex++];
			CHECK(cudaMemcpyAsync(deviceBuffers[bufferIndex], pipelineImages[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice, copyStream));
			CHECK(cudaEventRecord(done, copyStream));
			addNode(Node{ imageIndex, 0, bufferIndex, done });
		}
		if (kernelEventIndex != kKernelCount || !hasPendingNode[kReductionLevels - 1]) { fprintf(stderr, "Incomplete image reduction tree.\n"); std::exit(EXIT_FAILURE); }
		const Node finalNode = pendingNodes[kReductionLevels - 1];
		CHECK(cudaStreamWaitEvent(copyStream, finalNode.ready, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceBuffers[finalNode.bufferIndex], imageSize, cudaMemcpyDeviceToHost, copyStream));
		CHECK(cudaStreamSynchronize(copyStream));
		for (cudaEvent_t event : kernelDoneEvents) CHECK(cudaEventDestroy(event));
		for (cudaEvent_t event : copyDoneEvents) CHECK(cudaEventDestroy(event));
		CHECK(cudaStreamDestroy(computeStream)); CHECK(cudaStreamDestroy(copyStream));
		for (unsigned char* buffer : deviceBuffers) CHECK(cudaFree(buffer));

	};

	// 固定申请一片连续锁页内存：16 个输入槽和 1 个输出槽。
	// 申请与释放不进入每轮复用性能计时；普通内存到锁页内存的拷贝仍在循环内计时。
	unsigned char* pinnedStaging = nullptr;
	const auto pinnedAllocationStart = std::chrono::steady_clock::now();
	CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinnedStaging), pinnedStagingSize));
	const double pinnedAllocationTimeMs = std::chrono::duration<double, std::milli>(
		std::chrono::steady_clock::now() - pinnedAllocationStart).count();
	std::vector<cv::Mat> pinnedInputs;
	pinnedInputs.reserve(kImageCount);
	for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
		pinnedInputs.emplace_back(images[0].rows, images[0].cols, CV_8UC1,
			pinnedStaging + static_cast<size_t>(imageIndex) * imageSize);
	}
	cv::Mat pinnedOutput(images[0].rows, images[0].cols, CV_8UC1,
		pinnedStaging + static_cast<size_t>(kImageCount) * imageSize);

	for (int run = 0; run < totalRuns; ++run)
	{
		const auto start = std::chrono::steady_clock::now();

		// 普通内存到锁页 staging buffer 的 CPU 拷贝
		const auto stagingCopyStart = std::chrono::steady_clock::now();
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
			std::memcpy(pinnedInputs[imageIndex].ptr(), images[imageIndex].ptr(), imageSize);
		}
		stagingCopyTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - stagingCopyStart).count();

		runPipeline(pinnedInputs, pinnedOutput);

		// D2H 已在 helper 内同步完成；复制回普通 cv::Mat 以保留既有调用接口和校验流程。
		cv::Mat result(images[0].rows, images[0].cols, CV_8UC1);
		std::memcpy(result.ptr(), pinnedOutput.ptr(), imageSize);

		if (run == totalRuns - 1)
			blendingImage = result;

		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("GPU fixed pinned-host pipeline run %d: %.3f ms%s\n", run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}
	CHECK(cudaFreeHost(pinnedStaging));

	double measuredTimeSumMs = 0.0;
	double measuredStagingCopyTimeSumMs = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		measuredTimeSumMs += runTimesMs[run];
	for (int run = warmupRuns; run < totalRuns; ++run)
		measuredStagingCopyTimeSumMs += stagingCopyTimesMs[run];
	printf("GPU fixed pinned-host pipeline average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, measuredTimeSumMs / (totalRuns - warmupRuns));
	printf("GPU fixed pinned-host allocation: %.3f ms\n", pinnedAllocationTimeMs);
	printf("GPU fixed pinned-host staging copy average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, measuredStagingCopyTimeSumMs / (totalRuns - warmupRuns));

	return true;
}

// 核函数并发测试。
bool Proc7(std::vector<cv::Mat>& images, cv::Mat& blendingImage)
{
	if (images.empty() || images[0].empty() || images[0].type() != CV_8UC1 ||
		images[0].rows == 0 || images[0].cols == 0 || !images[0].isContinuous() ||
		images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (images[imageIndex].empty() || images[imageIndex].size() != images[0].size() ||
			images[imageIndex].type() != CV_8UC1 || !images[imageIndex].isContinuous())
			return false;

	const size_t imageSize = images[0].total();
	constexpr int threadsPerBlock = 256, totalRuns = 10, warmupRuns = 5;
	const int blendBlocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	double runTimesMs[totalRuns] = {};
	std::array<unsigned long long, kImageCount> finalSums = {};
	std::array<unsigned long long, kImageCount> finalSquaredSums = {};

	for (int run = 0; run < totalRuns; ++run) {
		const auto start = std::chrono::steady_clock::now();

		cv::Mat result(images[0].rows, images[0].cols, CV_8UC1);
		std::array<unsigned long long, kImageCount> hostSums = {};
		std::array<unsigned long long, kImageCount> hostSquaredSums = {};
		unsigned char* deviceImages = nullptr;
		unsigned char* deviceQualityImages = nullptr;
		unsigned long long* deviceSums = nullptr;
		unsigned long long* deviceSquaredSums = nullptr;
		cudaStream_t uploadStream = nullptr;
		cudaStream_t blendStream = nullptr;
		cudaStream_t qualityStream = nullptr;
		cudaStream_t downloadStream = nullptr;
		cudaEvent_t uploadsReady = nullptr;
		cudaEvent_t blendDone = nullptr;

		for (cv::Mat& image : images)
			CHECK(cudaHostRegister(image.ptr(), imageSize, cudaHostRegisterDefault));
		CHECK(cudaHostRegister(result.ptr(), imageSize, cudaHostRegisterDefault));

		CHECK(cudaMalloc(&deviceImages, imageSize * kImageCount));
		CHECK(cudaMalloc(&deviceQualityImages, imageSize * kImageCount));
		CHECK(cudaMalloc(&deviceSums, sizeof(unsigned long long) * kImageCount));
		CHECK(cudaMalloc(&deviceSquaredSums, sizeof(unsigned long long) * kImageCount));

		CHECK(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&blendStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&qualityStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&downloadStream, cudaStreamNonBlocking));

		CHECK(cudaEventCreateWithFlags(&uploadsReady, cudaEventDisableTiming));
		CHECK(cudaEventCreateWithFlags(&blendDone, cudaEventDisableTiming));

		// H2D数据传输
		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex)
			CHECK(cudaMemcpyAsync(deviceImages + static_cast<size_t>(imageIndex) * imageSize,
				images[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice, uploadStream));
		// 融合会原地覆盖左节点；质量检测读取独立快照，避免跨 stream 的读写竞争。
		CHECK(cudaMemcpyAsync(deviceQualityImages, deviceImages, imageSize * kImageCount,
			cudaMemcpyDeviceToDevice, uploadStream));
		CHECK(cudaEventRecord(uploadsReady, uploadStream));
		CHECK(cudaStreamWaitEvent(qualityStream, uploadsReady, 0));
		CHECK(cudaStreamWaitEvent(blendStream, uploadsReady, 0));

		// 先提交低占用质量检测：每个 SM 留出大部分 resident block 槽位给融合流。
		int smCount = 0;
		CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
		CHECK(cudaMemsetAsync(deviceSums, 0, sizeof(unsigned long long) * kImageCount, qualityStream));
		CHECK(cudaMemsetAsync(deviceSquaredSums, 0, sizeof(unsigned long long) * kImageCount, qualityStream));
		proc7InputQuality<<<smCount, threadsPerBlock,
			static_cast<size_t>(threadsPerBlock) * 2 * sizeof(unsigned long long), qualityStream>>>(
			deviceQualityImages, imageSize, deviceSums, deviceSquaredSums);
		CHECK(cudaGetLastError());

		int pairDistance = 1;
		while (pairDistance < kImageCount) {
			for (int firstImage = 0; firstImage + pairDistance < kImageCount;
				firstImage += pairDistance * 2) {
				unsigned char* left = deviceImages + static_cast<size_t>(firstImage) * imageSize;
				unsigned char* right = deviceImages + static_cast<size_t>(firstImage + pairDistance) * imageSize;
				blendImage<<<blendBlocksPerGrid, threadsPerBlock, 0, blendStream>>>(left, right, left, imageSize);
				CHECK(cudaGetLastError());
			}
			pairDistance <<= 1;
		}
		CHECK(cudaEventRecord(blendDone, blendStream));
		CHECK(cudaStreamWaitEvent(downloadStream, blendDone, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceImages, imageSize, cudaMemcpyDeviceToHost, downloadStream));
		CHECK(cudaMemcpyAsync(hostSums.data(), deviceSums, sizeof(unsigned long long) * kImageCount,
			cudaMemcpyDeviceToHost, qualityStream));
		CHECK(cudaMemcpyAsync(hostSquaredSums.data(), deviceSquaredSums,
			sizeof(unsigned long long) * kImageCount, cudaMemcpyDeviceToHost, qualityStream));
		CHECK(cudaStreamSynchronize(downloadStream));
		CHECK(cudaStreamSynchronize(qualityStream));

		CHECK(cudaEventDestroy(blendDone));
		CHECK(cudaEventDestroy(uploadsReady));
		CHECK(cudaStreamDestroy(downloadStream));
		CHECK(cudaStreamDestroy(qualityStream));
		CHECK(cudaStreamDestroy(blendStream));
		CHECK(cudaStreamDestroy(uploadStream));
		CHECK(cudaFree(deviceSquaredSums));
		CHECK(cudaFree(deviceSums));
		CHECK(cudaFree(deviceQualityImages));
		CHECK(cudaFree(deviceImages));
		CHECK(cudaHostUnregister(result.ptr()));
		for (cv::Mat& image : images)
			CHECK(cudaHostUnregister(image.ptr()));

		if (run == totalRuns - 1) {
			blendingImage = result;
			finalSums = hostSums;
			finalSquaredSums = hostSquaredSums;
		}
		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("GPU fusion + input-quality run %d: %.3f ms%s\n", run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}

	double total = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		total += runTimesMs[run];
	printf("GPU fusion + input-quality average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, total / (totalRuns - warmupRuns));

	double minimumMean = 255.0, maximumMean = 0.0;
	for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
		const double sampleCount = static_cast<double>(imageSize);
		const double mean = static_cast<double>(finalSums[imageIndex]) / sampleCount;
		const double variance = std::max(0.0, static_cast<double>(finalSquaredSums[imageIndex]) / sampleCount - mean * mean);
		const double standardDeviation = std::sqrt(variance);
		minimumMean = std::min(minimumMean, mean);
		maximumMean = std::max(maximumMean, mean);
		printf("Input %d quality: mean = %.2f, standard deviation = %.2f\n", imageIndex, mean, standardDeviation);
	}
	printf("Input brightness spread: %.2f; equal-weight fusion is %s.\n", maximumMean - minimumMean,
		maximumMean - minimumMean <= 10.0 ? "consistent" : "potentially exposure-sensitive");
	return true;
}

// 核函数满SM串行，对比Proc7
bool Proc7_Compare(std::vector<cv::Mat>& images, cv::Mat& blendingImage)
{
	if (images.empty() || images[0].empty() || images[0].type() != CV_8UC1 ||
		images[0].rows == 0 || images[0].cols == 0 || !images[0].isContinuous() ||
		images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
		if (images[imageIndex].empty() || images[imageIndex].size() != images[0].size() ||
			images[imageIndex].type() != CV_8UC1 || !images[imageIndex].isContinuous())
			return false;

	const size_t imageSize = images[0].total();
	constexpr int threadsPerBlock = 256, totalRuns = 10, warmupRuns = 5;
	const int blendBlocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	const size_t qualitySharedMemoryBytes = static_cast<size_t>(threadsPerBlock) * 2 * sizeof(unsigned long long);
	double runTimesMs[totalRuns] = {};
	std::array<unsigned long long, kImageCount> finalSums = {};
	std::array<unsigned long long, kImageCount> finalSquaredSums = {};

	for (int run = 0; run < totalRuns; ++run) {
		const auto start = std::chrono::steady_clock::now();
		cv::Mat result(images[0].rows, images[0].cols, CV_8UC1);
		std::array<unsigned long long, kImageCount> hostSums = {};
		std::array<unsigned long long, kImageCount> hostSquaredSums = {};
		unsigned char* deviceImages = nullptr;
		unsigned char* deviceQualityImages = nullptr;
		unsigned long long* deviceSums = nullptr;
		unsigned long long* deviceSquaredSums = nullptr;
		cudaStream_t uploadStream = nullptr;
		cudaStream_t blendStream = nullptr;
		cudaStream_t qualityStream = nullptr;
		cudaStream_t downloadStream = nullptr;
		cudaEvent_t uploadsReady = nullptr;
		cudaEvent_t blendDone = nullptr;

		for (cv::Mat& image : images)
			CHECK(cudaHostRegister(image.ptr(), imageSize, cudaHostRegisterDefault));
		CHECK(cudaHostRegister(result.ptr(), imageSize, cudaHostRegisterDefault));
		CHECK(cudaMalloc(&deviceImages, imageSize * kImageCount));
		CHECK(cudaMalloc(&deviceQualityImages, imageSize * kImageCount));
		CHECK(cudaMalloc(&deviceSums, sizeof(unsigned long long) * kImageCount));
		CHECK(cudaMalloc(&deviceSquaredSums, sizeof(unsigned long long) * kImageCount));
		CHECK(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&blendStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&qualityStream, cudaStreamNonBlocking));
		CHECK(cudaStreamCreateWithFlags(&downloadStream, cudaStreamNonBlocking));
		CHECK(cudaEventCreateWithFlags(&uploadsReady, cudaEventDisableTiming));
		CHECK(cudaEventCreateWithFlags(&blendDone, cudaEventDisableTiming));

		for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex)
			CHECK(cudaMemcpyAsync(deviceImages + static_cast<size_t>(imageIndex) * imageSize,
				images[imageIndex].ptr(), imageSize, cudaMemcpyHostToDevice, uploadStream));
		CHECK(cudaMemcpyAsync(deviceQualityImages, deviceImages, imageSize * kImageCount,
			cudaMemcpyDeviceToDevice, uploadStream));
		CHECK(cudaEventRecord(uploadsReady, uploadStream));
		CHECK(cudaStreamWaitEvent(qualityStream, uploadsReady, 0));
		CHECK(cudaStreamWaitEvent(blendStream, uploadsReady, 0));

		// 与 Proc7 使用同一质量 kernel；唯一改变是完整 grid 使该 kernel 可填满 SM。
		int smCount = 0;
		CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
		CHECK(cudaMemsetAsync(deviceSums, 0, sizeof(unsigned long long) * kImageCount, qualityStream));
		CHECK(cudaMemsetAsync(deviceSquaredSums, 0, sizeof(unsigned long long) * kImageCount, qualityStream));
		proc7InputQuality<<<blendBlocksPerGrid, threadsPerBlock, qualitySharedMemoryBytes, qualityStream>>>(
			deviceQualityImages, imageSize, deviceSums, deviceSquaredSums);
		CHECK(cudaGetLastError());

		// 同一 qualityStream 内，质量统计完成后才允许融合 kernel 开始。
		int pairDistance = 1;
		while (pairDistance < kImageCount) {
			for (int firstImage = 0; firstImage + pairDistance < kImageCount;
				firstImage += pairDistance * 2) {
				unsigned char* left = deviceImages + static_cast<size_t>(firstImage) * imageSize;
				unsigned char* right = deviceImages + static_cast<size_t>(firstImage + pairDistance) * imageSize;
				blendImage<<<blendBlocksPerGrid, threadsPerBlock, 0, qualityStream>>>(left, right, left, imageSize);
				CHECK(cudaGetLastError());
			}
			pairDistance <<= 1;
		}
		CHECK(cudaEventRecord(blendDone, qualityStream));
		CHECK(cudaStreamWaitEvent(downloadStream, blendDone, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceImages, imageSize, cudaMemcpyDeviceToHost, downloadStream));
		CHECK(cudaMemcpyAsync(hostSums.data(), deviceSums, sizeof(unsigned long long) * kImageCount,
			cudaMemcpyDeviceToHost, qualityStream));
		CHECK(cudaMemcpyAsync(hostSquaredSums.data(), deviceSquaredSums,
			sizeof(unsigned long long) * kImageCount, cudaMemcpyDeviceToHost, qualityStream));
		CHECK(cudaStreamSynchronize(downloadStream));
		CHECK(cudaStreamSynchronize(qualityStream));

		CHECK(cudaEventDestroy(blendDone));
		CHECK(cudaEventDestroy(uploadsReady));
		CHECK(cudaStreamDestroy(downloadStream));
		CHECK(cudaStreamDestroy(qualityStream));
		CHECK(cudaStreamDestroy(blendStream));
		CHECK(cudaStreamDestroy(uploadStream));
		CHECK(cudaFree(deviceSquaredSums));
		CHECK(cudaFree(deviceSums));
		CHECK(cudaFree(deviceQualityImages));
		CHECK(cudaFree(deviceImages));
		CHECK(cudaHostUnregister(result.ptr()));
		for (cv::Mat& image : images)
			CHECK(cudaHostUnregister(image.ptr()));

		if (run == totalRuns - 1) {
			blendingImage = result;
			finalSums = hostSums;
			finalSquaredSums = hostSquaredSums;
		}
		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("GPU full-grid quality + serial fusion run %d: %.3f ms%s\n", run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}

	double total = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		total += runTimesMs[run];
	printf("GPU full-grid quality + serial fusion average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, total / (totalRuns - warmupRuns));

	double minimumMean = 255.0, maximumMean = 0.0;
	for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex) {
		const double sampleCount = static_cast<double>(imageSize);
		const double mean = static_cast<double>(finalSums[imageIndex]) / sampleCount;
		const double variance = std::max(0.0, static_cast<double>(finalSquaredSums[imageIndex]) / sampleCount - mean * mean);
		const double standardDeviation = std::sqrt(variance);
		minimumMean = std::min(minimumMean, mean);
		maximumMean = std::max(maximumMean, mean);
		printf("Input %d quality: mean = %.2f, standard deviation = %.2f\n", imageIndex, mean, standardDeviation);
	}
	printf("Input brightness spread: %.2f; equal-weight fusion is %s.\n", maximumMean - minimumMean,
		maximumMean - minimumMean <= 10.0 ? "consistent" : "potentially exposure-sensitive");
	return true;
}
