#if (defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__) || (defined(__APPLE__) & defined(__MACH__)))
#include <cv.h>
#include <highgui.h>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "RatioCalculator.h"
#include <tesseract/baseapi.h>
#include <math.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "RatioCalculator.h"
#include <tesseract/baseapi.h>
#include <math.h>
#endif

#ifndef LICENSEPLATEREADER_H
#define LICENSEPLATEREADER_H

class LicensePlateReader{
public:
	LicensePlateReader();
	// ilker's variables
	double thresholding_value;
	int blur_value;
	cv::RNG rng;
	std::vector<cv::Rect> possiblePlates;
	RatioCalculator *rc;
	int erosion_value;
	int dilation_value;
	int blur_amount;
	int clean_brush_size;
	/// Global variables

	cv::Mat srcOriginal, srcModified, srcGray, srcCropped;
	cv::Mat dst, detected_edges;

	int edgeThresh = 1;
	int lowThreshold;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	char* window_name = "Edge Map";

	/* Function prototypes */
	void showDFTImage(cv::Mat&);
	void cleanImage(cv::Mat&, int);
	void calculateFilterValues(cv::Mat&);
	void drawSkewAngle(cv::Mat&);
	int readWithTesseract(cv::Mat&, int&, char*&);
	void convertBGR2RGB(cv::Mat& src, cv::Mat& dst);
	/* Function prototypes */

	/* Main Function*/
	char* readLicensePlates(const char* picturePath, cv::Mat& markedRGB, cv::Mat& lpRGB);
};

#endif