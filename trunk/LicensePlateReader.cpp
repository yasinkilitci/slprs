#include "LicensePlateReader.h"

using namespace std;
using namespace cv;

LicensePlateReader::LicensePlateReader()
{
	rng(12345);
	prepareTesseract();
}

char* LicensePlateReader::readLicensePlates(const char* picturePath, Mat& markedRGB, Mat& lpRGB)
{
	// Load the picture
	srcOriginal = imread(picturePath);

	/* Initialize Calculator */
	rc = new RatioCalculator();

	/* Calculate Thresholding Value */
	//GaussianBlur(srcOriginal,srcOriginal,Size(3,3),0,0);
	thresholding_value = rc->calculateThresholdValue(srcOriginal);
	//cout << "Threshold Value: " << thresholding_value << endl;

	/* Pre-Processing Stage */

	srcModified = srcOriginal.clone();
	cvtColor(srcModified, srcModified, CV_BGR2GRAY);
	//GaussianBlur(srcModified,srcModified,Size(3,3),0,0);
	//equalizeHist(srcModified,srcModified);
	threshold(srcModified, srcModified, thresholding_value + 25, 255, 0);
	//adaptiveThreshold(srcModified,srcModified,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3,5);

	srcGray = srcModified.clone();

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(srcOriginal.size(), srcOriginal.type());

	//////////////////////////////////////////////////////////////////////
	///////////////// CANNY THRESHOLD ///////////////////////////////////
	//////////////////////////////////////////////////////////////////////

	lowThreshold = 99;
	/// Canny detector
	Canny(srcGray, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	srcOriginal.copyTo(dst, detected_edges);
	//imshow( window_name, dst );

	/* Instead of Showing image, we continue from here and detect edges again */

	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat grayagain;
	cvtColor(dst, grayagain, CV_BGR2GRAY);
	/// Detect edges using Threshold
	threshold(grayagain, threshold_output, 1, 255, THRESH_BINARY);
	// dilate edges for filling gaps
	//imshow("Before Dilation", threshold_output);
	int krsize = 2;
	Mat mtemp;

	morphologyEx(threshold_output, mtemp, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(2 * krsize + 1, 2 * krsize + 1), Point(1, 1)));
	morphologyEx(mtemp, mtemp, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(2 * 5 + 1, 2 * 5 + 1), Point(1, 1)));
	//morphologyEx(threshold_output, threshold_output, MORPH_BLACKHAT, getStructuringElement(MORPH_CROSS, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1)));
	//morphologyEx(threshold_output, threshold_output, MORPH_DILATE, getStructuringElement(MORPH_CROSS, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1)));
	bitwise_not(mtemp, mtemp);
	bitwise_and(threshold_output, mtemp, mtemp);
	//morphologyEx(threshold_output, threshold_output, MORPH_DILATE, getStructuringElement(MORPH_CROSS, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1)));
	//morphologyEx( threshold_output, threshold_output, MORPH_DILATE, getStructuringElement(MORPH_ELLIPSE,Size(2*1+1,2*1+1),Point(1,1)) );
	//morphologyEx( threshold_output, threshold_output, MORPH_ERODE, getStructuringElement(MORPH_ELLIPSE,Size(2*1+1,2*1+1),Point(1,1)) );

	//imshow("After Dilation: Temp", mtemp);
	//imshow("After Dilation:Threshold Output", threshold_output);
	/// Find contours
	//imshow("thresholdoutput before", threshold_output);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//imshow("thresholdoutput after", threshold_output);

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}


	/// Draw polygonal contour + bonding rects + circles
	//Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	Mat drawing = srcOriginal.clone();
	
	for( int i = 0; i< contours.size(); i++ )
	{
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	//drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
	//circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
	}

	imshow("Bounding Rects", drawing);

	// Get possible plates, draw on the image, find the biggest one
	//cout << "Image Height: " << srcOriginal.rows << endl;
	possiblePlates = rc->getPossiblePlates(boundRect, srcOriginal.rows);
	vector<PlatePack> successfulPlates;
	PlatePack t_successfulPlate;
	int numberoftries = possiblePlates.size();
	
	while (numberoftries>0)
	{
		drawing = srcOriginal.clone();
		for (int i = 0; i<possiblePlates.size(); i++)
			rectangle(drawing, possiblePlates[i].tl(), possiblePlates[i].br(), Scalar(0, 0, 255), 2, 8, 0);

		Rect biggestRect;
		int biggestRectIndex;
		biggestRect = rc->getBiggestRect(possiblePlates, biggestRectIndex);


		//Scalar color = Scalar(0, 255, 120);
		//rectangle(drawing, biggestRect.tl(), biggestRect.br(), color, 2, 8, 0);
		//
		///////////////////////////////
		///* SHOW THE MARKED PICTURE */
		///////////////////////////////

		////namedWindow("Contours", CV_WINDOW_AUTOSIZE);
		////imshow("Contours", drawing);
		//Mat markedBGR = drawing.clone();
		//markedRGB = drawing.clone();
		//convertBGR2RGB(markedBGR, markedRGB);

		/* Crop the image from original */
		srcCropped = srcOriginal(biggestRect);

		processImage(srcCropped);

		////////////////////////////////////////
		////// DISCRETE FOURIER TRANSFORM //////
		////////////////////////////////////////
		//showDFTImage(white);

		///////////////////////////////////////////
		////// READ ITERATION WITH TESSERACT //////
		///////////////////////////////////////////
		int confidence = 0;
		char* plate_str;
		int plate_str_len = readWithTesseract(srcCropped, confidence, plate_str);
		//cout << "Length of plate is: " << plate_str_len << endl;
		if (plate_str_len < 11 && plate_str_len > 7 && confidence > 40)
		{
			// collect successful plates in vector
			t_successfulPlate.confidence = confidence;
			t_successfulPlate.plate = biggestRect;
			successfulPlates.push_back(t_successfulPlate);
			numberoftries--;
			possiblePlates.erase(possiblePlates.begin() + biggestRectIndex);
		}
		else
		{
			numberoftries--;
			possiblePlates.erase(possiblePlates.begin() + biggestRectIndex);
		}

	}

	/* Now we have only the best results in the possiblePlates list */

	////////////////////////////////////////////////////
	////////////   CHARACTER SEGMENTATION   ////////////
	////////////////////////////////////////////////////

	// first, we have to find the plate with the biggest confidence value

	Rect bestRect;
	int bestConfidence = 0;

	for (int i = 0; i < successfulPlates.size(); i++)
	{
		if (successfulPlates[i].confidence > bestConfidence)
		{
			bestConfidence = successfulPlates[i].confidence;
			bestRect = successfulPlates[i].plate;
		}
	}

	// Mark the Picture
	Scalar color = Scalar(0, 255, 120);
	rectangle(drawing, bestRect.tl(), bestRect.br(), color, 2, 8, 0);

	/////////////////////////////
	/* SHOW THE MARKED PICTURE */
	/////////////////////////////

	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", drawing);
	Mat markedBGR = drawing.clone();
	markedRGB = drawing.clone();
	convertBGR2RGB(markedBGR, markedRGB);

	// Crop with the best Rect
	srcCropped = srcOriginal(bestRect);

	processImage(srcCropped);

	//////////////////////////////
	/* SET THE CROPPED PICTURE */
	/////////////////////////////
	//imshow("Kesilmis Resim", srcCropped);
	Mat lpBGR = srcCropped.clone();
	cvtColor(srcCropped, lpBGR, CV_GRAY2BGR);
	lpRGB = lpBGR.clone();
	convertBGR2RGB(lpBGR, lpRGB);

	// find contours for the best rect, yes! These are the characters themselves!

	vector<vector<Point> > plate_contours;
	vector<Vec4i> plate_hierarchy;

	Mat plate_detected_edges;
	lowThreshold = 99;
	/// Canny detector
	Canny(srcCropped, plate_detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	/// Using Canny's output as a mask, we display our result

	findContours(plate_detected_edges, plate_contours, plate_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > plate_contours_poly(plate_contours.size());
	vector<Rect> plate_boundRect(plate_contours.size());

	for (int i = 0; i < plate_contours.size(); i++)
	{
		approxPolyDP(Mat(plate_contours[i]), plate_contours_poly[i], 3, true);
		plate_boundRect[i] = boundingRect(Mat(plate_contours_poly[i]));
	}

	/*draw contours*/

	Mat plate_drawing = srcCropped.clone();
	cvtColor(plate_drawing, plate_drawing, CV_GRAY2BGR);
	
	for( int i = 0; i< plate_contours.size(); i++ )
	{
		Scalar color = Scalar( 0, 255, 0 );
		drawContours( plate_drawing, plate_contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		rectangle( plate_drawing, plate_boundRect[i].tl(), plate_boundRect[i].br(), color, 2, 8, 0 );
	}

	imshow("Plate Contours", plate_drawing);
	
	
	//////////////////////////////////////
	// Copy characters onto white area
	//////////////////////////////////////
	Mat white = Mat::zeros( srcCropped.size(), CV_8UC3 );
	cvtColor(white,white,CV_BGR2GRAY);
	bitwise_not(white, white);
	//imshow("white", white);
	//srcCropped.copyTo(white(biggestRect));

	/* Tesseract - Read Single Char */

	STRING text("");
	for (int i = 0; i < plate_boundRect.size(); i++)
	{
		Mat m_singlechar = srcCropped(plate_boundRect[i]);
		if (m_singlechar.rows > srcCropped.rows / 2 && m_singlechar.rows < 9 * srcCropped.rows / 10)
		{
			// copy the character onto white area
			m_singlechar.copyTo(white(plate_boundRect[i]));
		}
	}

	imshow("white_concat", white);

	char* buffer;
	int confidence;
	readWithTesseract(white, confidence, buffer);
	text += buffer;

	char* result = new char[text.length() + 1];
	strncpy(result, text.string(), text.length() + 1);

	return result;
}

void LicensePlateReader::showDFTImage(Mat& I)
{
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).
	/* Post Processing */

	//GaussianBlur(magI,magI,Size(2*2+1,2*2+1),0,0);
	//adaptiveThreshold(magI,magI,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,CV_THRESH_BINARY,0,0);
	//threshold(magI,magI,1,255,CV_THRESH_BINARY);
	//bitwise_not(magI,magI);
	//erode(magI,magI,getStructuringElement(MORPH_RECT,Size(2*erosion_value+1,2*erosion_value+1),Point(erosion_value,erosion_value)));
	//dilate(magI,magI,getStructuringElement(MORPH_RECT,Size(4*dilation_value+1,4*dilation_value+1),Point(dilation_value,dilation_value)));
	//bitwise_not(magI,magI);

	drawSkewAngle(magI);

	imshow("Spectrum Magnitude", magI);

}

void LicensePlateReader::cleanImage(Mat& src, int border)
{
	rectangle(src,
		Point(0, 0),
		Point(src.cols, src.rows),
		Scalar(255, 255, 255),
		border,
		8);
}

void LicensePlateReader::calculateFilterValues(Mat& src)
{
	long pc = src.cols * src.rows;
	if (pc<10000)
	{
		erosion_value = 0;
		dilation_value = 0;
		blur_amount = 0;
		clean_brush_size = 1;
		//cout << "Category 1" << endl;
	}
	else if (pc >= 10000 && pc<20000)
	{
		erosion_value = 1;
		dilation_value = 1;
		blur_amount = 1;
		clean_brush_size = 5;
		//cout << "Category 2" << endl;
	}
	else if (pc >= 20000 && pc<30000)
	{
		erosion_value = 2;
		dilation_value = 1;
		blur_amount = 1;
		clean_brush_size = 20;
		//cout << "Category 3" << endl;
	}
	else if (pc >= 30000 && pc<40000)
	{
		erosion_value = 1;
		dilation_value = 1;
		blur_amount = 1;
		clean_brush_size = 20;
		//cout << "Category 4" << endl;
	}
	else if (pc >= 40000 && pc<50000)
	{
		erosion_value = 2;
		dilation_value = 2;
		blur_amount = 1;
		clean_brush_size = 15;
		//cout << "Category 5" << endl;
	}
	else if (pc >= 50000 && pc<100000)
	{
		erosion_value = 2;
		dilation_value = 2;
		blur_amount = 1;
		clean_brush_size = 30;
		//cout << "Category 6" << endl;
	}
	else if (pc >= 100000 && pc<150000)
	{
		erosion_value = 4;
		dilation_value = 3;
		blur_amount = 2;
		clean_brush_size = 60;
		//cout << "Category 7" << endl;
	}
	else if (pc >= 150000 && pc<200000)
	{
		erosion_value = 4;
		dilation_value = 4;
		blur_amount = 2;
		clean_brush_size = 80;
		//cout << "Category 8" << endl;
	}
	else if (pc >= 200000 && pc<250000)
	{
		erosion_value = 5;
		dilation_value = 4;
		blur_amount = 2;
		clean_brush_size = 100;
		//cout << "Category 9" << endl;
	}
	else
	{
		erosion_value = 5;
		dilation_value = 5;
		blur_amount = 2;
		clean_brush_size = 120;
		//cout << "Category 10" << endl;
	}
}

void LicensePlateReader::drawSkewAngle(Mat& dft)
{
	int countofwhites;
	double upperAvg = 0, lowerAvg = 0;
	int upperTotal = 0, lowerTotal = 0;
	int upperBrightestCol = 0, upperBrightestColIndex = 0, lowerBrightestCol = 0, lowerBrightestColIndex = 0;
	/* Operation Circle */
	int middleX = dft.cols / 2, middleY = dft.rows / 2;
	int upperRow = 4 * dft.rows / 10, lowerRow = 6 * dft.rows / 10;
	int gray = 0;

	/* Calculate average values of upper and lower guidelines */
	for (int col = 0; col<dft.cols; ++col)
	{
		gray = (int)dft.at<uchar>(upperRow, col);
		upperTotal += gray;
		if (gray>upperBrightestCol)
		{
			upperBrightestCol = gray;
			upperBrightestColIndex = col;
		}
		//cout<< "Gray is "<< gray  << endl;
	}

	for (int col = 0; col<dft.cols; ++col)
	{
		gray = (int)dft.at<uchar>(lowerRow, col);
		lowerTotal += gray;
		if (gray>lowerBrightestCol)
		{
			lowerBrightestCol = gray;
			lowerBrightestColIndex = col;
		}
		//cout<< "Gray is "<< gray  << endl;
	}

	upperAvg = upperTotal / dft.cols;
	lowerAvg = lowerTotal / dft.cols;

	cout << "Lower Average Value is " << lowerAvg << endl;
	cout << "Upper Average Value is " << upperAvg << endl;

	/* Draw Y Axis */
	line(dft, Point(middleX, 0),
		Point(middleX, dft.rows),
		Scalar(255, 255, 255),
		1,
		0);
	/* Draw X Axis */
	line(dft, Point(0, middleY),
		Point(dft.cols, middleY),
		Scalar(255, 255, 255),
		1,
		0);

	/* Draw Upper Line */
	line(dft, Point(0, upperRow),
		Point(dft.cols, upperRow),
		Scalar(255, 255, 255),
		1,
		0);

	/* Draw Lower Line */
	line(dft, Point(0, lowerRow),
		Point(dft.cols, lowerRow),
		Scalar(255, 255, 255),
		1,
		0);

	/* Draw Skew Line */
	line(dft, Point(upperBrightestColIndex, upperRow),
		Point(lowerBrightestColIndex, lowerRow),
		Scalar(255, 255, 255),
		1,
		0);

}

void LicensePlateReader::prepareTesseract()
{
	tess.Init(NULL, "lpa", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
}

int LicensePlateReader::readWithTesseract(Mat& srcPlate, int&confidence, char*& output)
{
	tess.SetImage((uchar*)srcPlate.data, srcPlate.cols, srcPlate.rows, 1, srcPlate.cols);

	// Get the text
	output = tess.GetUTF8Text();
	confidence = tess.MeanTextConf();
	/*
	std::cout << output << std::endl;
	cout << "Confidence Value is %" << confidence << endl;
	int* a = tess.AllWordConfidences();;
	while (*a != -1)
	{
		cout << "Confidence: " << *a << endl;
		a++;
	}
	*/
	// return count of characters
	return strlen(output) - 2;
}

void LicensePlateReader::convertBGR2RGB(Mat& src, Mat& dst)
{
	for (int row = 0; row < src.rows; ++row)
	for (int col = 0; col<src.cols; ++col)
	{
		dst.at<Vec3b>(row, col)[2] = src.at<Vec3b>(row, col)[0];
		dst.at<Vec3b>(row, col)[0] = src.at<Vec3b>(row, col)[2];
	}
}

void LicensePlateReader::processImage(Mat& image)
{
	/* Initialize filter values */
	calculateFilterValues(image);

	/* Process Cropped Image */

	thresholding_value = rc->calculateThresholdValue(image);
	//cout << "Threshold Value(Cropped): " << thresholding_value << endl;
	cvtColor(image, image, CV_BGR2GRAY);
	//equalizeHist(srcCropped,srcCropped);
	//medianBlur(srcCropped,srcCropped,2*blur_amount+1);
	threshold(image, image, thresholding_value, 255, 0);
	bitwise_not(image, image);
	erode(image, image, getStructuringElement(MORPH_RECT, Size(2 * erosion_value + 1, 2 * erosion_value + 1), Point(erosion_value, erosion_value)));
	dilate(image, image, getStructuringElement(MORPH_RECT, Size(2 * dilation_value + 1, 2 * dilation_value + 1), Point(dilation_value, dilation_value)));
	bitwise_not(image, image);
	//GaussianBlur(srcCropped,srcCropped,Size(2*blur_amount+1,2*blur_amount+1),0,0);
	medianBlur(image, image, 2 * blur_amount + 1);
	// Clean Image
	cleanImage(image, clean_brush_size);

}