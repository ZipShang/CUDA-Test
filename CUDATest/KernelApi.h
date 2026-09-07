#pragma once
#include <vector>
#include <opencv2/core.hpp>

bool Proc1(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc2(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
// Proc2 的线程块大小测试重载；前两个参数与基础融合流程保持一致。
bool Proc2(std::vector<cv::Mat>& images, cv::Mat& blendingImage, int ThreadNumPerBlock);
bool Proc3(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc4(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc5(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc5_Compare(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc6(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc7(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc7_Compare(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
