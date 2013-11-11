#include "RatioCalculator.h"
#include <cstddef>
#include <iostream>

using namespace std;
using namespace cv;

RatioCalculator::RatioCalculator()
{

}

vector<float> RatioCalculator::getRatioList(vector<Rect>& sourceRectList)
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

float RatioCalculator::getRatio(Rect& sourceRect)
{
	return sourceRect.width / sourceRect.height;
}

Rect RatioCalculator::getBiggestRect(vector<Rect>& sourceRectList, int &indexOfBiggest)
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
	//cout << "Height: " << srcMat.rows << endl;
	//cout << "Width: " << srcMat.cols << endl;
	int totalBlue = 0, totalGreen = 0, totalRed = 0;
	double average = 0;
	//double varianceBlue = 0, varianceGreen = 0, varianceRed = 0, variance = 0;
	int pixelCount = srcMat.cols * srcMat.rows;


	for (int row = 0; row < srcMat.rows; ++row)
		for (int col = 0; col<srcMat.cols; ++col)
		{
			totalBlue += (int)srcMat.at<Vec3b>(row, col)[0];
			totalGreen += (int)srcMat.at<Vec3b>(row, col)[1];
			totalRed += (int)srcMat.at<Vec3b>(row, col)[2];
		}

	average = (totalBlue + totalGreen + totalRed) / (3 * pixelCount);
	//cout << "Average Red Value: " << avgRed << endl;
	//cout << "Average Green Value: " << avgGreen << endl;
	//cout << "Average Blue Value: " << avgBlue << endl;

	/*for (int row = 0; row < srcMat.rows; ++row)
	for (int col = 0; col<srcMat.cols; ++col)
	{
		varianceBlue += pow((int)srcMat.at<Vec3b>(row, col)[0] - avgBlue, 2);
		varianceGreen += pow((int)srcMat.at<Vec3b>(row, col)[1] - avgGreen, 2);
		varianceRed += pow((int)srcMat.at<Vec3b>(row, col)[2] - avgRed, 2);
	}*/

	/*for (int row = 0; row < srcMat.rows; ++row)
	for (int col = 0; col<srcMat.cols; ++col)
	{
		variance += pow((int)srcMat.at<uchar>(row, col) - average, 2);
	}*/
	//variance = sqrt(variance / pixelCount);
	//cout << "Gray stddev : " << sqrt(variance / pixelCount) << endl;
	return average; //+ ((sqrt(varianceBlue/pixelCount)+sqrt(varianceGreen/pixelCount)+sqrt(varianceRed/pixelCount))/6);
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