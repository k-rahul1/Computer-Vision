//Created by Rahul Kumar

//Code to generate ORB feature in any target image

#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/features2d.hpp>

using std::string;

int main(){
    cv::VideoCapture vid;
    cv::Mat imgFrame;

    // Opening the webcam//Created by Rahul Kumar

//Declation of function which are used to implement different morphological filters for 2-D object recognition
    vid.open(0);
    //vid.open("http://10.0.0.26:4747/video");

    //Creating the windows to display video
    cv::namedWindow("ORB Feature", cv::WINDOW_NORMAL);

    // Checking for successful opening of video port
    if (!vid.isOpened()){
        std::cout << "Alert! Camera is not accessible" << std::endl;
        return -1;
    }

    int count = 1;

    while (true){
        vid.read(imgFrame);
        char key = cv::waitKey(2);

        //creating an object of ORB class 
        cv::Ptr<cv::ORB> orb = cv::ORB::create(200,1.2f,8,31,0,4,cv::ORB::HARRIS_SCORE,31,20);
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;

        //generating the keypoints and corresponding descriptors using ORB object
        orb->detectAndCompute(imgFrame,cv::noArray(),keyPoints,descriptors);

        cv::Mat featureImage;

        //drawing the ORB features
        cv::drawKeypoints(imgFrame,keyPoints,featureImage,cv::Scalar(0,0,255));

        string filename = "image" + std::to_string(count) + ".jpg";
        // Saving the images once 's' is pressed
        if(key =='s'){
            cv::imwrite(filename,featureImage);
            count++;
        }

        //displaying the video with ORB features 
        cv::resizeWindow("ORB Feature", cv::Size(720,720));
        cv::imshow("ORB Feature", featureImage);

        // Terminating the video if 'q' is pressed
        if (key == 'q'){
            std::cout<<"Terminating camera feed!"<<std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}