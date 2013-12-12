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

/* License plate categories by pixelcount*/
#define PLATE_SIZE_XXXS 1
#define PLATE_SIZE_XXS 2
#define PLATE_SIZE_XS 3
#define PLATE_SIZE_S 4
#define PLATE_SIZE_M 5
#define PLATE_SIZE_L 6
#define PLATE_SIZE_XL 7
#define PLATE_SIZE_XXL 8
#define PLATE_SIZE_XXXL 9
#define PLATE_SIZE_4XL 10
#define PLATE_SIZE_5XL 11

typedef struct PlatePack
{
	cv::Rect plate;
	int confidence;
} PlatePack;

class LicensePlateReader{
public:
	LicensePlateReader();
	/* Global Variables*/
	double thresholding_value;
	double colorbalance;
	cv::RNG rng;
	std::vector<cv::Rect> possiblePlates;
	RatioCalculator *rc;
	int erosion_value;
	int dilation_value;
	int blur_amount;
	int clean_brush_size;
	/* Global Variables*/

	cv::Mat srcOriginal, srcGray, srcCropped;
	cv::Mat dstGray, detected_edges;

	int canny_threshold;
	int canny_ratio;
	int canny_kernel_size;

	int morph_kernel_size;

	tesseract::TessBaseAPI tess;

	/* Function prototypes */
	void cleanImage(cv::Mat&, int);
	int calculateFilterValues(cv::Mat&);
	void prepareTesseract();
	int readWithTesseract(cv::Mat&, int&, char*&);
	int readWithTesseract(cv::Mat&, char*&);
	void convertBGR2RGB(cv::Mat& src, cv::Mat& dst);
	void processImage(cv::Mat& image);
	/* Function prototypes */

	/* Main Function*/
	char* readLicensePlates(const char* picturePath, cv::Mat& markedRGB, cv::Mat& lpRGB);
};

#endif