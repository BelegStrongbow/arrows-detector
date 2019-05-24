#include "pch.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

void imageToImage(const Mat &srcMat, Mat &dstMat, const vector<Point2f> &dst) {
	vector<Point2f> src;
	src.push_back(Point2f(0, 0));
	src.push_back(Point2f(dstMat.cols, 0));
	src.push_back(Point2f(dstMat.cols, dstMat.rows));
	src.push_back(Point2f(0, dstMat.rows));
	Mat warp_matrix = getPerspectiveTransform(dst, src);
	warpPerspective(srcMat, dstMat, warp_matrix, Size(dstMat.cols, dstMat.rows));
}

MatND histogram (Mat hsv, Mat mask, int hbins = 30, int sbins = 32)
{
	int histSize[] = { hbins, sbins };
	// hue varies from 0 to 179, see cvtColor
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	MatND hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = { 0, 1 };
	calcHist(&hsv, 1, channels, mask,
		hist, 2, histSize, ranges,
		true, // the histogram is uniform
		false);
	return hist;
}

void viewHist (MatND hist, int hbins=30, int sbins=32)
{
	double maxVal = 0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int scale = 10;
	Mat histImg = Mat::zeros(sbins*scale, hbins * 10, CV_8UC3);
	for (int h = 0; h < hbins; h++)
		for (int s = 0; s < sbins; s++)
		{
			float binVal = hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(histImg, Point(h*scale, s*scale),
				Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				Scalar::all(intensity),
				-1);
		}
	namedWindow("H-S Histogram", 1);
	imshow("H-S Histogram", histImg);
	waitKey();
}

vector<Point2f> searchForRectangle(Mat src)
{
	Mat gray, mask;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, mask, 0, 255, THRESH_BINARY | THRESH_OTSU); //for later - to detect the rectangle
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(5, 5),
		Point(2, 2));
	morphologyEx(mask, mask, MORPH_CLOSE, element);
	morphologyEx(mask, mask, MORPH_OPEN, element);
	medianBlur(gray, gray, 5);
	vector<vector<Point>> contours; //starting to detect rectangle - finding contours
	vector<Vec4i> hierarchy;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int biggestContourIdx = -1;
	float biggestContourArea = 0;
	Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 100, 0);
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());

		float ctArea = contourArea(contours[i]);
		if (ctArea > biggestContourArea)
		{
			biggestContourArea = ctArea;
			biggestContourIdx = i;
		}
	}
	RotatedRect boundingBox = minAreaRect(contours[biggestContourIdx]);
	vector<Point2f> corners(4);
	boundingBox.points(corners.data());
	line(drawing, corners[0], corners[1], Scalar(255, 255, 255));
	line(drawing, corners[1], corners[2], Scalar(255, 255, 255));
	line(drawing, corners[2], corners[3], Scalar(255, 255, 255));
	line(drawing, corners[3], corners[0], Scalar(255, 255, 255));
	imshow("drawing", drawing);
	return corners;

}

vector<Mat> createMasks(Mat tar)
{
	int center[] = { tar.rows / 2, tar.cols / 2 };
	int d = tar.rows / 10.5;
	Mat e[10];
	vector<Mat> maskTar(10);
	for (int i = 0; i < 10; i++)
	{
		e[i] = getStructuringElement(MORPH_ELLIPSE,
			Size(d*(i + 1), d*(i + 1)),
			Point(d / 2, d / 2));
		maskTar[i].push_back(Mat::zeros({ 300, 300 }, e[i].type()));
		e[i].copyTo(maskTar[i](Rect(center[0] - d * (i + 1) / 2, center[1] - d * (i + 1) / 2, d*(i + 1), d*(i + 1))));

		for (int j = 0; j < i; j++)
			maskTar[i] = maskTar[i] - maskTar[j];
	}
	Mat result = Mat::zeros({ 300, 300 }, tar.type());
	tar.copyTo(result, maskTar[9]);
	return maskTar;
}

Mat searchForArrows(Mat src)
{
	auto corners = searchForRectangle(src);

	imshow("input", src); //aand displays the rectangle


	Mat tar = Mat::zeros({ 300, 300 }, src.type());
	imageToImage(src, tar, corners);
	imshow("t", tar);

	Mat hsv;
	cvtColor(tar, hsv, COLOR_BGR2HSV); //convert to HSV


/*	vector <Mat> tarHSV;
	split(hsv, tarHSV); //divide into three separate channels

	imshow("h", tarHSV[0]);
	imshow("s", tarHSV[1]);
	imshow("v", tarHSV[2]);

	Mat rotated = tarHSV[0].clone();
	rotate(rotated, rotated, ROTATE_90_CLOCKWISE);
	absdiff(rotated, tarHSV[0], rotated);
	imshow("czysty", rotated);*/
	auto maskTar = createMasks(tar);
	Mat arrows = Mat::zeros({ 300, 300 }, maskTar[9].type());

	int ranges[5][4] = { {65, 46, 65, 54},
						{10, 10, 20, 20},
						{80, 16, 80, 54},
						{40, 30, 50, 50},
						{80, 16, 80, 54} };

	for (int i = 0; i < 10; i++)
	{
		Point p;
		MatND hist = histogram(hsv, maskTar[i].clone());
		viewHist(hist);
		minMaxLoc(hist, 0, 0, 0, &p);
		Mat maskRange;
		inRange(hsv, Scalar(p.y * 180 / 30 - ranges[i / 2][0], p.x * 256 / 32 - ranges[i / 2][1], 0), Scalar(p.y * 180 / 30 + ranges[i / 2][2], p.x * 256 / 32 + ranges[i / 2][3], 256), maskRange);
		Mat invertedMask, maskFinal;
		bitwise_not(maskRange, invertedMask);
		Mat arrowsFrag = Mat::zeros({ 300, 300 }, tar.type());
		Mat maskTarErode;
		//erode(maskTar[i], maskTarErode, element);
		invertedMask.copyTo(arrowsFrag, maskTar[i]);
		imshow("arrowsFrag", arrowsFrag);
		arrows = arrows + arrowsFrag;

	}
	return arrows;
}

Mat markArrows(Mat src)
{
	Mat arrows = searchForArrows(src);
	imshow("jhgghjh",arrows);
	Mat elementBig = getStructuringElement(MORPH_ELLIPSE,
		Size(9, 9),
		Point(4, 4));
	Mat arrowsErode;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(5, 5),
		Point(2, 2));
	erode(arrows, arrows, element);
	return arrows;
}

int goalFunction(Mat src)
{
	Mat arrows = searchForArrows(src);
	int amountOfWhite = countNonZero(src);
	return amountOfWhite;
}

int main(int argc, char** argv)
{
	const char* filename = argc >= 2 ? argv[1] : "Target.jpg";
	// Loads an image
	Mat src = imread("Target5.jpg", IMREAD_COLOR);
	Size size(300, src.rows*300/src.cols);
	resize(src, src, size);
	
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_name -- default %s] \n", filename);
		return -1;
	}
	Mat arrows = markArrows(src);
	imshow("maska", arrows);
	waitKey(0);
	return 0;
}