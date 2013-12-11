#ifndef RATIOCALCULATOR_H
#define RATIOCALCULATOR_H
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



#define ERROR_AMOUNT 1.0f

/* LICENCE PLATE RATIOS */

/* 11 x 52 cm Standard */
#define RATIO_TR_STD 4.73f
/* 15 x 24 cm For Motorbikes */
#define RATIO_TR_MIN 1.6f

/* LICENCE PLATE RATIOS */

// min originalwidth/platewidth value
#define MIN_PLATE_HEIGHT_RATIO 25.0f


// COLOR WEIGHT STATES
#define STATE_COLOR_BLUE 1
#define STATE_COLOR_RED 2
#define STATE_COLOR_NONE 0

///////////////////////////////////////////
///////// LICENSE PLATE DIMENSIONS ////////
///////////////////////////////////////////

#define DIM_LP_TR_HEIGHT 11
#define DIM_LP_TR_WIDTH 52

class RatioCalculator
{
public:
	// FUNCTIONS
	RatioCalculator();
	std::vector<float> getRatioList(std::vector<cv::Rect> sourceRectList);
	float getRatio(cv::Rect sourceRect);
	cv::Rect getBiggestRect(std::vector<cv::Rect> sourceRectList, int &indexOfBiggest);
	std::vector<cv::Rect> getPossiblePlates(std::vector<cv::Rect> &RectList, int imageHeight);
	double calculateThresholdValue(cv::Mat&);
	double calculateThresholdValue(cv::Mat& srcMat, int& weight);
	void convertBGR2RGB(cv::Mat& src, cv::Mat& dst);
	// VARIABLES
private:
	std::vector<cv::Rect> possiblePlates;
};

#endif