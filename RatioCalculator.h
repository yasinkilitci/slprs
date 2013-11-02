#ifndef RATIOCALCULATOR_H
#define RATIOCALCULATOR_H
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



#define ERROR_AMOUNT 0.9f

/* LICENCE PLATE RATIOS */

/* 11 x 52 cm Standard */
#define RATIO_TR_STD 4.73f
/* 15 x 24 cm For Motorbikes */
#define RATIO_TR_MIN 1.6f

/* LICENCE PLATE RATIOS */

// min originalwidth/platewidth value
#define MIN_PLATE_HEIGHT_RATIO 25.0f

class RatioCalculator
{
public:
	// FUNCTIONS
	RatioCalculator();
	std::vector<float> getRatioList(std::vector<cv::Rect> sourceRectList);
	float getRatio(cv::Rect sourceRect);
	cv::Rect getBiggestRect(std::vector<cv::Rect> sourceRectList, int &indexOfBiggest);
	std::vector<cv::Rect> getPossiblePlates(std::vector<cv::Rect> &RectList,int imageHeight);
	double calculateThresholdValue(cv::Mat&);
	// VARIABLES
private:
	std::vector<cv::Rect> possiblePlates;
};

#endif