#include "RatioCalculator.h"
#include <cstddef>
#include <iostream>

using namespace std;
using namespace cv;

RatioCalculator::RatioCalculator()
{

}

vector<float> RatioCalculator::getRatioList(vector<Rect> sourceRectList)
{
	vector<float> ratioVector;
	float currentRatio;
	for (int i = 0; i<sourceRectList.size(); i++)
	{
		currentRatio = sourceRectList[i].width / sourceRectList[i].height;
		ratioVector.push_back(currentRatio);
	}

	return ratioVector;
}

float RatioCalculator::getRatio(Rect sourceRect)
{
	return sourceRect.width / sourceRect.height;
}

Rect RatioCalculator::getBiggestRect(vector<Rect> sourceRectList, int &indexOfBiggest)
{
	cv::Rect tempRect;
	if (sourceRectList.size() == 0)
		return tempRect;

	int biggestIndex = 0;
	int biggestSize = 0;
	for (int i = 0; i<sourceRectList.size(); i++)
	{
		if (sourceRectList[i].width*sourceRectList[i].height > biggestSize)
		{
			biggestSize = sourceRectList[i].width*sourceRectList[i].height;
			biggestIndex = i;
		}
	}

	indexOfBiggest = biggestIndex;
	return sourceRectList[biggestIndex];
}

vector<Rect> RatioCalculator::getPossiblePlates(vector<Rect> &RectList, int imageHeight)
{
	vector<Rect> possiblePlates;
	float minpratio = 0.0f;

	for (int i = 0; i< RectList.size(); i++)
	{
		float ratio = (float)RectList[i].width / RectList[i].height;
		if (ratio > RATIO_TR_STD - ERROR_AMOUNT &&
			ratio < RATIO_TR_STD + ERROR_AMOUNT &&
			((float)(imageHeight / RectList[i].height) < MIN_PLATE_HEIGHT_RATIO)
			)
		{
			possiblePlates.push_back(RectList[i]);
		}
	}
	return possiblePlates;
}

double RatioCalculator::calculateThresholdValue(Mat &srcMat)
{
	int totalBlue = 0, totalGreen = 0, totalRed = 0;
	double avgBlue = 0, avgGreen = 0, avgRed = 0, average = 0;
	int pixelCount = srcMat.cols * srcMat.rows;


	for (int row = 0; row < srcMat.rows; ++row)
	for (int col = 0; col<srcMat.cols; ++col)
	{
		totalBlue += (int)srcMat.at<Vec3b>(row, col)[0];
		totalGreen += (int)srcMat.at<Vec3b>(row, col)[1];
		totalRed += (int)srcMat.at<Vec3b>(row, col)[2];
	}

	avgBlue = totalBlue / pixelCount;
	avgGreen = totalGreen / pixelCount;
	avgRed = totalRed / pixelCount;
	average = (avgBlue + avgGreen + avgRed) / 3;
	
	return average;
}

double RatioCalculator::calculateThresholdValue(Mat& srcMat, int& weight)
{
	int totalBlue = 0, totalGreen = 0, totalRed = 0;
	double avgBlue = 0, avgGreen = 0, avgRed = 0, average = 0;
	int pixelCount = srcMat.cols * srcMat.rows;

	for (int row = 0; row < srcMat.rows; ++row)
	for (int col = 0; col<srcMat.cols; ++col)
	{
		totalBlue += (int)srcMat.at<Vec3b>(row, col)[0];
		totalGreen += (int)srcMat.at<Vec3b>(row, col)[1];
		totalRed += (int)srcMat.at<Vec3b>(row, col)[2];
	}

	avgBlue = totalBlue / pixelCount;
	avgGreen = totalGreen / pixelCount;
	avgRed = totalRed / pixelCount;

	average = (avgBlue + avgGreen + avgRed) / 3;

	weight = (avgBlue >= (avgGreen + avgRed)*0.8) ? (STATE_COLOR_BLUE) :
		(avgRed >= (avgGreen + avgBlue)*0.8) ? STATE_COLOR_RED : STATE_COLOR_NONE;


	return average;
}

double RatioCalculator::calculateThresholdValue(Mat& srcMat, double& colorbalance)
{
	int totalBlue = 0, totalGreen = 0, totalRed = 0;
	double avgBlue = 0, avgGreen = 0, avgRed = 0, average = 0;
	int pixelCount = srcMat.cols * srcMat.rows;


	for (int row = 0; row < srcMat.rows; ++row)
	for (int col = 0; col<srcMat.cols; ++col)
	{
		totalBlue += (int)srcMat.at<Vec3b>(row, col)[0];
		totalGreen += (int)srcMat.at<Vec3b>(row, col)[1];
		totalRed += (int)srcMat.at<Vec3b>(row, col)[2];
	}

	avgBlue = totalBlue / pixelCount;
	avgGreen = totalGreen / pixelCount;
	avgRed = totalRed / pixelCount;
	average = (avgBlue + avgGreen + avgRed) / 3;
	colorbalance = (avgRed / avgBlue) / (avgRed / avgGreen) / (avgBlue / avgGreen);

	return average;
}

void RatioCalculator::convertBGR2RGB(Mat& src, Mat& dst)
{
	for (int row = 0; row < src.rows; ++row)
		for (int col = 0; col<src.cols; ++col)
		{
			dst.at<Vec3b>(row, col)[2] = src.at<Vec3b>(row, col)[0];
			dst.at<Vec3b>(row, col)[0] = src.at<Vec3b>(row, col)[2];
		}
}