//Created by Rahul Kumar

//Code to find and draw corners of chessboard pattern

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include"function.h"

using std::vector;

//Function to find the corners of chessboard pattern
int findCorner(cv::Mat &image, vector<cv::Point2f> &corner_set, bool &patternFound){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Size patternSize(9, 6);
    patternFound = cv::findChessboardCorners(image, patternSize, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
    
    //generating the corners of chessboard
    if (patternFound){
        cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.0001));
    }
    
    //drawing the line between chessboard corners
    cv::drawChessboardCorners(image, patternSize, corner_set, patternFound);

    return 0;
}