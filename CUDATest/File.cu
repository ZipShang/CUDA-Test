#include <stdio.h>
#include <cstdlib>
#include <array>
#include <chrono>
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

namespace {

constexpr int kImageCount = 16;
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

// compute stream & copy stream,实现最大程度的显存复用，数据传输与核函数调用并行执行
namespace {

bool ValidateAsyncPipelineInputs(const std::vector<cv::Mat>& images)
{
	if (images.empty() || images[0].empty() || images[0].type() != CV_8UC1 ||
		images[0].rows == 0 || images[0].cols == 0 || !images[0].isContinuous())
		return false;
	if (images.size() != kImageCount)
		return false;

	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex) {
		if (images[imageIndex].empty() || images[imageIndex].cols != images[0].cols ||
			images[imageIndex].rows != images[0].rows ||
			images[imageIndex].type() != CV_8UC1 || !images[imageIndex].isContinuous())
			return false;
	}
	return true;
}

void RunAsyncPipelineOnce(std::vector<cv::Mat>& images, cv::Mat& result,
	size_t imageSize, int blocksPerGrid, int threadsPerBlock)
{
	result.create(images[0].rows, images[0].cols, CV_8UC1);
	std::array<unsigned char*, kDeviceBufferCount> deviceBuffers = {};
	std::array<cudaEvent_t, kImageCount> copyDoneEvents = {};
	std::array<cudaEvent_t, kKernelCount> kernelDoneEvents = {};
	cudaStream_t copyStream = nullptr;
	cudaStream_t computeStream = nullptr;

	// 注册页锁定 Host 内存，使 cudaMemcpyAsync 可以与计算重叠。
	for (cv::Mat& image : images)
		CHECK(cudaHostRegister(image.ptr(), imageSize, cudaHostRegisterDefault));
	CHECK(cudaHostRegister(result.ptr(), imageSize, cudaHostRegisterDefault));

	for (unsigned char*& deviceBuffer : deviceBuffers)
		CHECK(cudaMalloc(&deviceBuffer, imageSize));
	for (cudaEvent_t& event : copyDoneEvents)
		CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
	for (cudaEvent_t& event : kernelDoneEvents)
		CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
	CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
	CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

	std::array<Node, kReductionLevels> pendingNodes = {};
	std::array<bool, kReductionLevels> hasPendingNode = {};
	std::array<ReusableBuffer, kDeviceBufferCount> reusableBuffers = {};
	int reusableBufferCount = kDeviceBufferCount;
	int copyEventIndex = 0;
	int kernelEventIndex = 0;

	for (int bufferIndex = 0; bufferIndex < kDeviceBufferCount; ++bufferIndex)
		reusableBuffers[bufferIndex] = { bufferIndex, nullptr };

	auto releaseBuffer = [&](int bufferIndex, cudaEvent_t reusableAfter) {
		if (reusableBufferCount >= kDeviceBufferCount) {
			fprintf(stderr, "Buffer pool overflow.\n");
			std::exit(EXIT_FAILURE);
		}
		reusableBuffers[reusableBufferCount++] = { bufferIndex, reusableAfter };
	};

	auto acquireBuffer = [&]() {
		if (reusableBufferCount == 0) {
			fprintf(stderr, "Buffer pool exhausted.\n");
			std::exit(EXIT_FAILURE);
		}

		int reusableBufferIndex = 0;
		for (int index = 0; index < reusableBufferCount; ++index) {
			if (reusableBuffers[index].reusableAfter == nullptr) {
				reusableBufferIndex = index;
				break;
			}
		}

		const ReusableBuffer buffer = reusableBuffers[reusableBufferIndex];
		reusableBuffers[reusableBufferIndex] = reusableBuffers[--reusableBufferCount];
		if (buffer.reusableAfter != nullptr)
			CHECK(cudaStreamWaitEvent(copyStream, buffer.reusableAfter, 0));
		return buffer.bufferIndex;
	};

	auto mergeNodes = [&](const Node& left, const Node& right) {
		if (left.level != right.level ||
			left.firstImage + (1 << left.level) != right.firstImage ||
			kernelEventIndex >= kKernelCount) {
			fprintf(stderr, "Invalid image reduction tree.\n");
			std::exit(EXIT_FAILURE);
		}

		CHECK(cudaStreamWaitEvent(computeStream, left.ready, 0));
		CHECK(cudaStreamWaitEvent(computeStream, right.ready, 0));
		blendImage<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(
			deviceBuffers[left.bufferIndex], deviceBuffers[right.bufferIndex],
			deviceBuffers[left.bufferIndex], imageSize);
		CHECK(cudaGetLastError());

		cudaEvent_t kernelDone = kernelDoneEvents[kernelEventIndex++];
		CHECK(cudaEventRecord(kernelDone, computeStream));
		releaseBuffer(right.bufferIndex, kernelDone);
		return Node{ left.firstImage, left.level + 1, left.bufferIndex, kernelDone };
	};

	auto addNode = [&](Node node) {
		while (hasPendingNode[node.level]) {
			const Node left = pendingNodes[node.level];
			hasPendingNode[node.level] = false;
			node = mergeNodes(left, node);
		}
		pendingNodes[node.level] = node;
		hasPendingNode[node.level] = true;
	};

	for (int imageIndex = 0; imageIndex < kImageCount; ++imageIndex)
	{
		const int bufferIndex = acquireBuffer();
		cudaEvent_t copyDone = copyDoneEvents[copyEventIndex++];
		CHECK(cudaMemcpyAsync(deviceBuffers[bufferIndex], images[imageIndex].ptr(), imageSize,
			cudaMemcpyHostToDevice, copyStream));
		CHECK(cudaEventRecord(copyDone, copyStream));
		addNode(Node{ imageIndex, 0, bufferIndex, copyDone });
	}

	if (kernelEventIndex != kKernelCount || !hasPendingNode[kReductionLevels - 1]) {
		fprintf(stderr, "Incomplete image reduction tree.\n");
		std::exit(EXIT_FAILURE);
	}

	const Node finalNode = pendingNodes[kReductionLevels - 1];
	CHECK(cudaStreamWaitEvent(copyStream, finalNode.ready, 0));
	CHECK(cudaMemcpyAsync(result.ptr(), deviceBuffers[finalNode.bufferIndex], imageSize,
		cudaMemcpyDeviceToHost, copyStream));
	CHECK(cudaStreamSynchronize(copyStream));

	for (cudaEvent_t event : kernelDoneEvents)
		CHECK(cudaEventDestroy(event));
	for (cudaEvent_t event : copyDoneEvents)
		CHECK(cudaEventDestroy(event));
	CHECK(cudaStreamDestroy(computeStream));
	CHECK(cudaStreamDestroy(copyStream));
	for (unsigned char* deviceBuffer : deviceBuffers)
		CHECK(cudaFree(deviceBuffer));
	CHECK(cudaHostUnregister(result.ptr()));
	for (cv::Mat& image : images)
		CHECK(cudaHostUnregister(image.ptr()));
}

constexpr int kConcurrentPipelineCount = 3;

enum class PipelineExecutionMode {
	Single,
	ConcurrentHostThreads,
	SerialHostThread,
};

std::vector<cv::Mat> CloneImages(const std::vector<cv::Mat>& images)
{
	std::vector<cv::Mat> copies;
	copies.reserve(images.size());
	for (const cv::Mat& image : images)
		copies.push_back(image.clone());
	return copies;
}

bool RunAsyncPipelineBenchmark(std::vector<cv::Mat>& images, cv::Mat& blendingImage,
	PipelineExecutionMode executionMode, const char* benchmarkName)
{
	if (!ValidateAsyncPipelineInputs(images))
		return false;

	const size_t imageSize = images[0].total();
	constexpr int threadsPerBlock = 256;
	const int blocksPerGrid = static_cast<int>((imageSize + threadsPerBlock - 1) / threadsPerBlock);
	constexpr int totalRuns = 10;
	constexpr int warmupRuns = 5;
	double runTimesMs[totalRuns] = {};
	const int pipelineCount = executionMode == PipelineExecutionMode::Single ? 1 :
		kConcurrentPipelineCount;
	std::array<std::vector<cv::Mat>, kConcurrentPipelineCount> pipelineInputs;
	if (executionMode == PipelineExecutionMode::Single) {
		pipelineInputs[0] = images;
	}
	else {
		for (int pipelineIndex = 0; pipelineIndex < pipelineCount; ++pipelineIndex)
			pipelineInputs[pipelineIndex] = CloneImages(images);
	}

	for (int run = 0; run < totalRuns; ++run)
	{
		const auto start = std::chrono::steady_clock::now();
		std::array<cv::Mat, kConcurrentPipelineCount> pipelineResults;

		if (executionMode == PipelineExecutionMode::ConcurrentHostThreads) {
			std::array<std::thread, kConcurrentPipelineCount> pipelineThreads;
			for (int pipelineIndex = 0; pipelineIndex < pipelineCount; ++pipelineIndex) {
				pipelineThreads[pipelineIndex] = std::thread([&, pipelineIndex] {
					RunAsyncPipelineOnce(pipelineInputs[pipelineIndex], pipelineResults[pipelineIndex],
						imageSize, blocksPerGrid, threadsPerBlock);
				});
			}
			for (std::thread& pipelineThread : pipelineThreads)
				pipelineThread.join();
		}
		else {
			for (int pipelineIndex = 0; pipelineIndex < pipelineCount; ++pipelineIndex) {
				RunAsyncPipelineOnce(pipelineInputs[pipelineIndex], pipelineResults[pipelineIndex],
					imageSize, blocksPerGrid, threadsPerBlock);
			}
		}

		if (run == totalRuns - 1)
			blendingImage = pipelineResults[0];

		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("%s run %d: %.3f ms%s\n", benchmarkName, run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}

	double measuredTimeSumMs = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		measuredTimeSumMs += runTimesMs[run];
	printf("%s average (runs %d-%d): %.3f ms\n", benchmarkName, warmupRuns + 1,
		totalRuns, measuredTimeSumMs / (totalRuns - warmupRuns));

	return true;
}

} // namespace

// compute stream & copy stream,最大化显存复用，将图像传输流与核函数调用流最大程度并行
bool Proc3(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	return RunAsyncPipelineBenchmark(Images, BlendingImage, PipelineExecutionMode::Single,
		"GPU async pipeline");
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
	return RunAsyncPipelineBenchmark(Images, BlendingImage,
		PipelineExecutionMode::ConcurrentHostThreads,
		"Three host threads + GPU async pipelines");
}

// 在调用线程串行提交与 Proc5 数量相同的 GPU 流水线。
bool Proc5_Compare(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	return RunAsyncPipelineBenchmark(Images, BlendingImage,
		PipelineExecutionMode::SerialHostThread,
		"Single host thread + GPU async pipelines");
}


