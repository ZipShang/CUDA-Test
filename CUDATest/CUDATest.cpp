#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "KernelApi.h"

using namespace std;

namespace {

// Mirrors Proc's pairwise queue reduction and its integer truncation exactly.
bool ProcCPU(const vector<cv::Mat>& images, cv::Mat& blendingImage)
{
	if (images.empty() || images[0].empty() || images[0].type() != CV_8UC1 ||
		!images[0].isContinuous())
		return false;

	const cv::Size imageSize = images[0].size();
	const size_t pixelCount = images[0].total();
	queue<cv::Mat> workQueue;

	for (const cv::Mat& image : images) {
		if (image.empty() || image.type() != CV_8UC1 || image.size() != imageSize ||
			!image.isContinuous())
			return false;
		workQueue.push(image.clone());
	}

	while (workQueue.size() > 1) {
		cv::Mat src1 = workQueue.front();
		workQueue.pop();
		cv::Mat src2 = workQueue.front();
		workQueue.pop();
		cv::Mat dst(imageSize, CV_8UC1);

		const unsigned char* src1Data = src1.ptr<unsigned char>();
		const unsigned char* src2Data = src2.ptr<unsigned char>();
		unsigned char* dstData = dst.ptr<unsigned char>();
		for (size_t i = 0; i < pixelCount; ++i) {
			// Match the GPU kernel: integer promotion followed by division that rounds down.
			dstData[i] = static_cast<unsigned char>((src1Data[i] + src2Data[i]) / 2);
		}

		workQueue.push(std::move(dst));
	}

	blendingImage = workQueue.front();
	return true;
}

bool CompareImages(const cv::Mat& cpuImage, const cv::Mat& gpuImage)
{
	if (cpuImage.empty() || gpuImage.empty() || cpuImage.type() != gpuImage.type() ||
		cpuImage.size() != gpuImage.size())
		return false;

	cv::Mat difference;
	cv::compare(cpuImage, gpuImage, difference, cv::CMP_NE);
	const int mismatchedPixels = cv::countNonZero(difference);
	if (mismatchedPixels == 0) {
		cout << "CPU and GPU results match." << endl;
		return true;
	}

	cv::Mat absoluteDifference;
	cv::absdiff(cpuImage, gpuImage, absoluteDifference);
	double maxDifference = 0.0;
	cv::minMaxLoc(absoluteDifference, nullptr, &maxDifference);
	cerr << "CPU/GPU mismatch: " << mismatchedPixels
		<< " pixels differ; maximum difference = " << maxDifference << endl;
	return false;
}

} // namespace

int main()
{
	const int nElem = 16;
	const string InputPath = "C:\\image\\GoldenMatch\\CaptureImage\\Product1\\CUDATest";
	const string OutputPath = "C:\\image\\GoldenMatch\\CaptureImage\\Product1\\BlendingImage.bmp";
	vector<string> ImagePath;
	vector<cv::Mat> Images;
	ImagePath.reserve(nElem);
	Images.reserve(nElem);
	cv::glob(InputPath, ImagePath);
	if (ImagePath.size() != static_cast<size_t>(nElem))
	{
		cerr << "Input image count error." << endl;
		return 1;
	}

	for (int i = 0; i < nElem; i++)
	{
		Images.push_back(cv::imread(ImagePath[i], cv::IMREAD_GRAYSCALE));
	}

	cv::Mat cpuBlendingImage;
	if (!ProcCPU(Images, cpuBlendingImage))
	{
		cerr << "CPU image blending failed." << endl;
		return 1;
	}

	cv::Mat gpuBlendingImage;
	if (!Proc4(Images, gpuBlendingImage))
	{
		cerr << "GPU image blending failed." << endl;
		return 1;
	}

	if (!CompareImages(cpuBlendingImage, gpuBlendingImage))
		return 1;

	if (!cv::imwrite(OutputPath, gpuBlendingImage))
	{
		cerr << "Unable to write output image." << endl;
		return 1;
	}

	cout << "GPU image blending succeeded and was verified by the CPU result." << endl;
	cout << "Press Enter to exit...";
	cin.get();
	return 0;
}
