//Created by Rahul Kumar

//Code for camera calibration using chessboard pattern

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include<vector>
#include "function.h"

using std::vector;
using std::string;

int main(){
    cv::VideoCapture vid;
    cv::Mat image;
    
    // Opening the video channel by streaming phone camera
    vid.open("http://10.0.0.26:4747/video");
    //vid.open(0);

    // Checking for successful opening of video port
    if (!vid.isOpened()){
        std::cout << "Alert! Camera is not accessible" << std::endl;
        return -1;
    }

    //Creating the windows to display video
    cv::namedWindow("pattern found", cv::WINDOW_NORMAL);
    int count=1;

    //declaring the vector to store pixel coordinates and world coordinates for set of images
    vector<vector<cv::Point2f>> corner_list;
    vector<vector<cv::Vec3f>> point_list;

    while(true){
        vid.read(image);
        char key = cv::waitKey(2);

        //checking for successful reading of image frame
        if(image.empty()){
            std::cout<<"warning! No frame available to display"<<std::endl;
            break;
        }

        //declaring the vector to store pixel coordinates and world coordinates for a single image
        vector<cv::Point2f> corner_set;
        vector<cv::Vec3f> point_set;

        bool patternFound;

        //finding the chessboard patter in the video feed
        findCorner(image,corner_set,patternFound);

        if(!patternFound){
            continue;
        }

        //providing the world coordinates corresponding to the pixel coordinates of chessboard corners
        for(int i=0;i>-6;i--){
            for(int j=0;j<9;j++){
                point_set.push_back(cv::Vec3f(j,i,0));
            }
        }

        string filename = "image" + std::to_string(count) + ".jpg";
        // Saving the images once 's' is pressed
        if(key =='s'){
            cv::imwrite(filename,image);

            //generating the pixel coordinates and corresponding world coordinates vectors
            corner_list.push_back(corner_set);
            point_list.push_back(point_set);
            
            //std::cout << "Numbers of corners list: " << corner_list.size()<<" "<<corner_list[0].size() << std::endl;
            //std::cout << "Numbers of point list: " << point_list.size() << " "<<point_list[0].size()<<std::endl;
            count++;
        }

        if(key == 'c'){
            //initializing the camera matrix
            cv::Mat camera_matrix = cv::Mat::eye(3,3,CV_64FC1);
            camera_matrix.at<double>(0,2) = image.cols/2;
            camera_matrix.at<double>(1,2) = image.rows/2;

            //initializing the distortion coefficient
            cv::Mat dist_coeff = cv::Mat::zeros(8,1,CV_64FC1);

            cv::Mat rot, trans;

            //displaying the initialized camera matrix and distortion coefficients before calibration
            std::cout<<"Camera matrix before calibration:"<<std::endl;
            std::cout<<camera_matrix<<std::endl;
            std::cout<<"Distortion coefficients before calibration:"<<std::endl;
            std::cout<<dist_coeff<<std::endl;

            //calibrating the camera
            double reprojError = cv::calibrateCamera(point_list,corner_list,image.size(),camera_matrix,dist_coeff,rot,trans,cv::CALIB_FIX_ASPECT_RATIO);

            //displaying the initialized camera matrix, distortion coefficients and reprojection error after calibration
            std::cout<<"Camera matrix after calibration:"<<std::endl;
            std::cout<<camera_matrix<<std::endl;
            std::cout<<"Distortion coefficients after calibration:"<<std::endl;
            std::cout<<dist_coeff<<std::endl;
            std::cout<<"Reprojection error: "<<reprojError<<std::endl;

            //storing the camera matrix and distortion coefficient in an xml file
            string filename = "Intrinsic_parameters.xml";
            cv::FileStorage fs(filename,cv::FileStorage::WRITE);
            fs<<"camera_matrix"<<camera_matrix;
            fs<<"distortion_coefficient"<<dist_coeff;
            fs.release();
        }
        
        // std::cout << "Numbers of corners found: " << corner_set.size() << std::endl;
        // std::cout << "X and Y position of 1st corner: " << corner_set[0].x << " " << corner_set[0].y << std::endl;
        // std::cout << "X and Y position of 1st corner: " << corner_set[1].x << " " << corner_set[1].y << std::endl;
        // std::cout << "X and Y position of 1st corner: " << corner_set[9].x << " " << corner_set[9].y << std::endl;
        
        //displaying the video feed
        cv::resizeWindow("pattern found", cv::Size(1280, 960));
        cv::imshow("pattern found", image);

        // Terminating the video if 'q' is pressed
        if (key == 'q'){
            std::cout<<"Terminating camera feed!"<<std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}