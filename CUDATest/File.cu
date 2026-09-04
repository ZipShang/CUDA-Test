#include <stdio.h>
#include <cstdlib>
#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
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

__global__ void blendImage(const unsigned char* src1,
	const unsigned char* src2, unsigned char* dst, size_t size)
{
	const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < size) {
		// uint8 values are promoted to int before addition, so the sum cannot overflow.
		dst[index] = static_cast<unsigned char>((src1[index] + src2[index]) / 2);
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

bool Proc3(std::vector<cv::Mat>& Images, cv::Mat& BlendingImage)
{
	// 在注册页锁定内存前，验证固定的 16 张灰度输入图像。
	if (Images.empty() || Images[0].empty() || Images[0].type() != CV_8UC1 ||
		Images[0].rows == 0 || Images[0].cols == 0 || !Images[0].isContinuous())
		return false;
	if (Images.size() != kImageCount)
		return false;
	for (int imageIndex = 1; imageIndex < kImageCount; ++imageIndex)
	{
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

		cv::Mat result(Images[0].rows, Images[0].cols, CV_8UC1);
		std::array<unsigned char*, kDeviceBufferCount> deviceBuffers = {};
		std::array<cudaEvent_t, kImageCount> copyDoneEvents = {};
		std::array<cudaEvent_t, kKernelCount> kernelDoneEvents = {};
		cudaStream_t copyStream = nullptr;
		cudaStream_t computeStream = nullptr;

		// 注册页锁定主机内存，使 cudaMemcpyAsync 可以与计算重叠。
		for (cv::Mat& image : Images)
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

		// 二进制进位式归约中，每一层最多暂存一个未配对节点。
		std::array<Node, kReductionLevels> pendingNodes = {};
		std::array<bool, kReductionLevels> hasPendingNode = {};
		std::array<ReusableBuffer, kDeviceBufferCount> reusableBuffers = {};
		int reusableBufferCount = kDeviceBufferCount;
		int copyEventIndex = 0;
		int kernelEventIndex = 0;

		for (int bufferIndex = 0; bufferIndex < kDeviceBufferCount; ++bufferIndex)
			reusableBuffers[bufferIndex] = { bufferIndex, nullptr };

		// buffer 被释放后，仍需等待最后一次 kernel 读取完成才能复用。
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

			// 将等待操作排入传输流；CPU 不阻塞，等 kernel 结束后立即复用该槽位。
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

			// 两个输入可能来自 H2D 或前一轮 kernel；统一等待各自的就绪事件。
			CHECK(cudaStreamWaitEvent(computeStream, left.ready, 0));
			CHECK(cudaStreamWaitEvent(computeStream, right.ready, 0));
			blendImage<<<blocksPerGrid, threadsPerBlock, 0, computeStream>>>(
				deviceBuffers[left.bufferIndex], deviceBuffers[right.bufferIndex],
				deviceBuffers[left.bufferIndex], imageSize);
			CHECK(cudaGetLastError());

			cudaEvent_t kernelDone = kernelDoneEvents[kernelEventIndex++];
			CHECK(cudaEventRecord(kernelDone, computeStream));
			// 结果原地保存在 left；right 待本次 kernel 读取完即可交还给传输流。
			releaseBuffer(right.bufferIndex, kernelDone);
			return Node{ left.firstImage, left.level + 1, left.bufferIndex, kernelDone };
		};

		auto addNode = [&](Node node) {
			// 二进制进位式归约：同层的相邻节点立即配对，严格保持原有融合顺序。
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
			// copyStream 只负责生产原图节点，完成后以 event 通知计算流。
			CHECK(cudaMemcpyAsync(deviceBuffers[bufferIndex], Images[imageIndex].ptr(), imageSize,
				cudaMemcpyHostToDevice, copyStream));
			CHECK(cudaEventRecord(copyDone, copyStream));
			addNode(Node{ imageIndex, 0, bufferIndex, copyDone });
		}

		if (kernelEventIndex != kKernelCount || !hasPendingNode[kReductionLevels - 1]) {
			fprintf(stderr, "Incomplete image reduction tree.\n");
			std::exit(EXIT_FAILURE);
		}

		const Node finalNode = pendingNodes[kReductionLevels - 1];
		// 最终 D2H 也放在传输流，等待最终归约节点生成后再开始。
		CHECK(cudaStreamWaitEvent(copyStream, finalNode.ready, 0));
		CHECK(cudaMemcpyAsync(result.ptr(), deviceBuffers[finalNode.bufferIndex], imageSize,
			cudaMemcpyDeviceToHost, copyStream));
		CHECK(cudaStreamSynchronize(copyStream));

		if (run == totalRuns - 1)
			BlendingImage = result;

		// 释放与解除页锁定同样属于本轮完整生命周期，必须在停止计时前完成。
		for (cudaEvent_t event : kernelDoneEvents)
			CHECK(cudaEventDestroy(event));
		for (cudaEvent_t event : copyDoneEvents)
			CHECK(cudaEventDestroy(event));
		CHECK(cudaStreamDestroy(computeStream));
		CHECK(cudaStreamDestroy(copyStream));
		for (unsigned char* deviceBuffer : deviceBuffers)
			CHECK(cudaFree(deviceBuffer));
		CHECK(cudaHostUnregister(result.ptr()));
		for (cv::Mat& image : Images)
			CHECK(cudaHostUnregister(image.ptr()));

		runTimesMs[run] = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		printf("GPU async pipeline run %d: %.3f ms%s\n", run + 1, runTimesMs[run],
			run < warmupRuns ? " (warm-up)" : "");
	}

	double measuredTimeSumMs = 0.0;
	for (int run = warmupRuns; run < totalRuns; ++run)
		measuredTimeSumMs += runTimesMs[run];
	printf("GPU async pipeline average (runs %d-%d): %.3f ms\n", warmupRuns + 1,
		totalRuns, measuredTimeSumMs / (totalRuns - warmupRuns));

	return true;
}

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
