#pragma once
#include <vector>
#include <opencv2/core.hpp>

bool Proc1(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc2(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc3(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc4(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc5(std::vector<cv::Mat>& images, cv::Mat& blendingImage);
bool Proc5_Compare(std::vector<cv::Mat>& images, cv::Mat& blendingImage);