#include "LicensePlateReader.h"

using namespace std;
using namespace cv;

LicensePlateReader::LicensePlateReader()
{
	/* random seed*/
	rng(12345);
	/* creates new instance of Tesseract API and sets the variables */
	prepareTesseract();
	/* Initialize Calculator */
	rc = new RatioCalculator();

	canny_threshold = 99;
	canny_ratio = 3;
	canny_kernel_size = 3;
	morph_kernel_size = 2;

	showBoundingRects = false;
	showMarkedPicture = false;
	useEqualizeHistogram = false;
	showChSeg = false;
	showCannyResult = false;
	showPlateContours = false;
}

char* LicensePlateReader::readLicensePlates(const char* picturePath, Mat& markedRGB, Mat& lpRGB)
{
	destroyAllWindows();
	// Load the picture
	srcOriginal = imread(picturePath);
	
	srcGray = srcOriginal.clone();

	//imshow("original", srcOriginal);

	/* Calculate Thresholding Value , Obtain the colorbalance value */
	//GaussianBlur(srcOriginal,srcOriginal,Size(3,3),0,0);
	medianBlur(srcOriginal, srcOriginal, 3);
	thresholding_value = calculateThresholdValue(srcOriginal,colorbalance);

	/* Pre-Processing Stage */

	cvtColor(srcGray, srcGray, CV_BGR2GRAY);


	//imshow("orgGray", srcGray);
	//GaussianBlur(srcModified,srcModified,Size(3,3),0,0);

	if (useEqualizeHistogram)
	{
		imshow("Histogram Eþitleme (Öncesi)", srcGray);
		equalizeHist(srcGray, srcGray);
		imshow("Histogram Eþitleme (Sonrasý)", srcGray);
		threshold(srcGray, srcGray, thresholding_value + 135, 255, 0);
	}
	else
		threshold(srcGray, srcGray, thresholding_value + 25, 255, 0);

	//adaptiveThreshold(srcModified,srcModified,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,3,5);
	//imshow("EqHist-Threshold", srcGray);

	/// Create a matrix of the same type and size as src (for dst)
	dstGray.create(srcOriginal.size(), srcOriginal.type());

	//////////////////////////////////////////////////////////////////////
	///////////////// CANNY THRESHOLD ///////////////////////////////////
	//////////////////////////////////////////////////////////////////////

	/// Canny detector
	Canny(srcGray, detected_edges, canny_threshold, canny_threshold*canny_ratio, canny_kernel_size);
	/// Using Canny's output as a mask, we display our result
	dstGray = Scalar::all(0);
	srcOriginal.copyTo(dstGray, detected_edges);

	if (showCannyResult)
		imshow( "Canny Image", dstGray );

	/* Instead of Showing image, we continue from here and detect edges again */
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//Mat grayagain;
	cvtColor(dstGray, dstGray, CV_BGR2GRAY);
	/// Detect edges using Threshold
	threshold(dstGray, dstGray, 1, 255, THRESH_BINARY);
	
	// dilate edges for filling gaps
	//imshow("Before Erode", dstGray);
	Mat mtemp;

	morphologyEx(dstGray, mtemp, MORPH_ERODE, getStructuringElement(MORPH_RECT, Size(2 * morph_kernel_size + 1, 2 * morph_kernel_size + 1), Point(1, 1)));
	//imshow("After Erode", mtemp);
	morphologyEx(mtemp, mtemp, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(2 * 5 + 1, 2 * 5 + 1), Point(1, 1)));
	//imshow("After Close", mtemp);
	bitwise_not(mtemp, mtemp);
	bitwise_and(dstGray, mtemp, mtemp);

	//imshow("Before Contour", dstGray);
	Mat drawing = dstGray.clone();
	findContours(dstGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	

	/// Approximate contours to polygons + get bounding rects
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	//vector<Point2f>center(contours.size());
	//vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		//minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}

	/// Draw polygonal contour + bonding rects + circles
	//Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	//Mat drawing = dstGray.clone();
	cvtColor(drawing, drawing, CV_GRAY2BGR);
	
	for( int i = 0; i< contours.size(); i++ )
	{
		//drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 2, 8, 0);
		//circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
	}

	// all the bounding rects on the screen
	if (showBoundingRects)
		imshow("Bounding Rects", drawing);
		

	// Get possible plates
	possiblePlates = rc->getPossiblePlates(boundRect, srcOriginal.rows);
	drawing = srcOriginal.clone();
	for (int i = 0; i<possiblePlates.size(); i++)
		rectangle(drawing, possiblePlates[i].tl(), possiblePlates[i].br(), Scalar(0, 0, 255), 2, 8, 0);

	if (showMarkedPicture)
		imshow("Plates", drawing);

	vector<PlatePack> successfulPlates;
	PlatePack t_successfulPlate;
	int numberoftries = possiblePlates.size();
	
	while (numberoftries>0)
	{
		
		Rect biggestRect;
		int biggestRectIndex;
		biggestRect = rc->getBiggestRect(possiblePlates, biggestRectIndex);

		/* Crop the image from original */
		srcCropped = srcOriginal(biggestRect);

		processImage(srcCropped);

		///////////////////////////////////////////
		////// READ ITERATION WITH TESSERACT //////
		///////////////////////////////////////////
		int confidence = 0;
		char* plate_str;
		int plate_str_len = readWithTesseract(srcCropped, confidence, plate_str);

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

	/* If at least one plate is successful, continue; otherwise send error message*/
	if (successfulPlates.size())
	{
		

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
		/// Canny detector
		Canny(srcCropped, plate_detected_edges, canny_threshold, canny_threshold*canny_ratio, canny_kernel_size);
		/// Using Canny's output as a mask, we display our result

		findContours(plate_detected_edges, plate_contours, plate_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<vector<Point> > plate_contours_poly(plate_contours.size());
		vector<Rect> plate_boundRect(plate_contours.size());

		for (int i = 0; i < plate_contours.size(); i++)
		{
			approxPolyDP(Mat(plate_contours[i]), plate_contours_poly[i], 3, true);
			plate_boundRect[i] = boundingRect(Mat(plate_contours_poly[i]));
		}

		/*Mark bounding rects for characters*/

		Mat plate_drawing = srcCropped.clone();
		cvtColor(plate_drawing, plate_drawing, CV_GRAY2BGR);

		for (int i = 0; i< plate_contours.size(); i++)
		{
			rectangle(plate_drawing, plate_boundRect[i].tl(), plate_boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
		}

		if (showPlateContours)
			imshow("Plate Contours", plate_drawing);


		//////////////////////////////////////
		// Copy characters onto white area
		//////////////////////////////////////
		Mat white = Mat::zeros(srcCropped.size(), CV_8UC3);
		cvtColor(white, white, CV_BGR2GRAY);
		bitwise_not(white, white);
	
		// color state shows us the color weight of the picture. blue / red / none
		int color_state = 0;
		/* Tesseract - Read Single Char */

		STRING text("");
		for (int i = 0; i < plate_boundRect.size(); i++)
		{
			Mat m_singlechar = srcCropped(plate_boundRect[i]);
			if (m_singlechar.rows > srcCropped.rows / 2 && m_singlechar.rows < 9 * srcCropped.rows / 10)
			{
				// if the character is not license plate band
				// copy the character onto white area

				/* set srcCropped from original Picture again (we calculate blue values over it*/
				// Crop plate from the original pic, then crop the character from the plate
				calculateThresholdValue((srcOriginal(bestRect))(plate_boundRect[i]), color_state);

				if (!color_state)
					m_singlechar.copyTo(white(plate_boundRect[i]));
			}
		}

		/* show the assembled picture*/
		if (showChSeg)
			imshow("white_concat", white);

		char* buffer;
		readWithTesseract(white, buffer);
		text += buffer;

		char* result = new char[text.length() + 1];
		strncpy(result, text.string(), text.length() + 1);

		return result;
	}
	else//read failed, set black pictures for both plate and pic
	{
		/////////////////////////////
		/* PICTURE OF FAILED READ  */
		/////////////////////////////

		Mat markedBGR = srcOriginal.clone();
		markedRGB = srcOriginal.clone();
		convertBGR2RGB(markedBGR, markedRGB);

		/////////////////////////////////
		/* PIC OF FAILED LICENSE PLATE */
		/////////////////////////////////

		lpRGB = Mat(Size(520, 110), CV_8UC3);
		lpRGB = Mat::zeros(lpRGB.size(), CV_8UC3);
		putText(lpRGB, "PLAKA OKUNAMIYOR",
			Point(50, 60),
			CV_FONT_HERSHEY_COMPLEX, 1.3, Scalar(255, 255, 255));
		
		return "Baþarýsýz Okuma!";
	}
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

int LicensePlateReader::calculateFilterValues(Mat& src)
{
	//pixel count
	long pc = src.cols * src.rows;
	if (pc<10000)
	{
		erosion_value = 0;
		dilation_value = 0;
		blur_amount = 0;
		clean_brush_size = 1;
		return PLATE_SIZE_XXXS;
	}
	else if (pc >= 10000 && pc<20000)
	{
		erosion_value = 1;
		dilation_value = 1;
		blur_amount = 1;
		clean_brush_size = 5;
		return PLATE_SIZE_XXS;
	}
	else if (pc >= 20000 && pc<30000)
	{
		erosion_value = 2;
		dilation_value = 1;
		blur_amount = 1;
		clean_brush_size = 20;
		return PLATE_SIZE_XS;
	}
	else if (pc >= 30000 && pc<40000)
	{
		erosion_value = 1;
		dilation_value = 1;
		blur_amount = 1;
		clean_brush_size = 20;
		return PLATE_SIZE_S;
	}
	else if (pc >= 40000 && pc<50000)
	{
		erosion_value = 2;
		dilation_value = 2;
		blur_amount = 1;
		clean_brush_size = 15;
		return PLATE_SIZE_M;
	}
	else if (pc >= 50000 && pc<100000)
	{
		erosion_value = 2;
		dilation_value = 2;
		blur_amount = 1;
		clean_brush_size = 30;
		return PLATE_SIZE_L;
	}
	else if (pc >= 100000 && pc<150000)
	{
		erosion_value = 4;
		dilation_value = 3;
		blur_amount = 2;
		clean_brush_size = 35;
		return PLATE_SIZE_XL;
	}
	else if (pc >= 150000 && pc<200000)
	{
		erosion_value = 4;
		dilation_value = 4;
		blur_amount = 2;
		clean_brush_size = 40;
		return PLATE_SIZE_XXL;
	}
	else if (pc >= 200000 && pc<250000)
	{
		erosion_value = 5;
		dilation_value = 4;
		blur_amount = 2;
		clean_brush_size = 45;
		return PLATE_SIZE_XXXL;
	}
	else
	{
		erosion_value = 5;
		dilation_value = 5;
		blur_amount = 2;
		clean_brush_size = 50;
		return PLATE_SIZE_4XL;
	}
}

void LicensePlateReader::prepareTesseract()
{
	tess.Init(NULL, "lpa", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ");
}

int LicensePlateReader::readWithTesseract(Mat& srcPlate, int&confidence, char*& output)
{
	//Set the image
	tess.SetImage((uchar*)srcPlate.data, srcPlate.cols, srcPlate.rows, 1, srcPlate.cols);

	// Get the text
	output = tess.GetUTF8Text();
	confidence = tess.MeanTextConf();
	// return count of characters
	return strlen(output) - 2;
}

int LicensePlateReader::readWithTesseract(Mat& srcPlate, char*& output)
{
	//Set the image
	tess.SetImage((uchar*)srcPlate.data, srcPlate.cols, srcPlate.rows, 1, srcPlate.cols);

	// Get the text
	output = tess.GetUTF8Text();
	
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

	thresholding_value = calculateThresholdValue(image);
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

void LicensePlateReader::setOptions(unsigned int options)
{
	showBoundingRects = ((options & SHOW_BOUNDING_RECTS) != 0) ? true : false;
	showMarkedPicture = ((options & SHOW_MARKED_PICTURE) != 0) ? true : false;
	useEqualizeHistogram = ((options & USE_EQUALIZE_HISTOGRAM) != 0) ? true : false;
	showChSeg = ((options & SHOW_CHARACTER_SEG) != 0) ?  true : false;
	showCannyResult = ((options & SHOW_CANNY_RESULT) != 0)? true:false;
	showPlateContours = ((options & SHOW_PLATE_CONTOURS) != 0) ? true : false;
}