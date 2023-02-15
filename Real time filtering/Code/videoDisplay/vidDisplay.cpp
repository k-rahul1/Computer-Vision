#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include "filter.h"

using std::string;

int main(){
    cv::VideoCapture vid;
    cv::Mat imgFrame;

    // Opening the channel 0(webcam) to display video
    vid.open(0);

    // Checking for successful opening of video port
    if(!vid.isOpened()){
        std::cout<<"Alert! Camera is not accessible"<<std::endl;
        return -1;
    }
    int i=1;
    std::cout<<"Press 'g' for grayscale video"<<std::endl;
    std::cout<<"Press 'h' for alternate grayscale video"<<std::endl;
    std::cout<<"Press 'b' for gaussian blur"<<std::endl;
    std::cout<<"Press 'x' for sobel X"<<std::endl;
    std::cout<<"Press 'y' for sobel Y"<<std::endl;
    std::cout<<"Press 'm' for gradient magnitude"<<std::endl;
    std::cout<<"Press 'i' for blur & quantize"<<std::endl;
    std::cout<<"Press 'c' for cartoonisation"<<std::endl;
    std::cout<<"Press 'n' for color negative"<<std::endl;
    std::cout<<"Press 'o' for original video"<<std::endl;
    std::cout<<"Press 's' to save image"<<std::endl;
    std::cout<<"Press 'q' to exit"<<std::endl;
    cv::namedWindow("LiveVideo",cv::WINDOW_NORMAL);
    char gray,anothergray,blur,sobelx,sobely,gradMag,blurquant,Cartoon,negative;

    // Creating the loop to display the video frame by frame
    while(true){
        vid.read(imgFrame);
        char key = cv::waitKey(2);
        if(key=='g'){
            // blur = key;
            // gray = key;
            // anothergray = key;
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative = key;
        }
        else if(key=='h'){
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative = key;
        }
        else if(key =='o'){
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative = key;
        }
        else if(key =='b'){
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        else if(key =='x'){
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        else if(key =='y'){
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        else if(key =='m'){
            blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        else if(key =='i'){
           blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        else if(key =='c'){
           blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        else if(key =='n'){
           blur = gray = anothergray = sobelx = sobely = gradMag = blurquant = Cartoon = negative =key;
        }
        
        if(gray=='g'){
            cv::cvtColor(imgFrame,imgFrame,cv::COLOR_BGR2GRAY);
        }
        else if(anothergray=='h'){
            cv::Mat grayFrame;
            greyscale(imgFrame,grayFrame);
            imgFrame = grayFrame;
        }
        else if(blur=='b'){
            cv::Mat blurFrame;
            blur5x5(imgFrame,blurFrame);
            imgFrame = blurFrame;
        }
        else if(sobelx=='x'){
            cv::Mat sobelxFrame;
            sobelX3x3(imgFrame,sobelxFrame);
            cv::convertScaleAbs(sobelxFrame,sobelxFrame);
            imgFrame = sobelxFrame;
        }
        else if(sobely=='y'){
            cv::Mat sobelyFrame;
            sobelY3x3(imgFrame,sobelyFrame);
            cv::convertScaleAbs(sobelyFrame,sobelyFrame);
            imgFrame = sobelyFrame;
        }
        else if(gradMag=='m'){
            cv::Mat sobelxFrame, sobelyFrame, gradMagFrame;
            sobelY3x3(imgFrame,sobelyFrame);
            sobelX3x3(imgFrame,sobelxFrame);
            magnitude(sobelxFrame,sobelyFrame, gradMagFrame);
            imgFrame = gradMagFrame;
        }
        else if(blurquant=='i'){
            cv::Mat quantFrame;
            blurQuantize(imgFrame,quantFrame,10);
            imgFrame = quantFrame;
        }
        else if(Cartoon=='c'){
            cv::Mat cartoonFrame;
            cartoon(imgFrame,cartoonFrame,15,15);
            imgFrame = cartoonFrame;
        }
        else if(negative=='n'){
            cv::Mat negativeFrame;
            Negative(imgFrame,negativeFrame);
            imgFrame = negativeFrame;
        }

        //checking for successful reading of image frame
        if(imgFrame.empty()){
            std::cout<<"warning! No frame available to display"<<std::endl;
            break;
        }
        
        string filename = "image" + std::to_string(i) + ".jpg";
        cv::resizeWindow("LiveVideo",cv::Size(1620,1620));
        cv::imshow("LiveVideo", imgFrame);
        //char key = cv::waitKey(5);

        // Saving the images once 's' is pressed
        if(key =='s'){
            cv::imwrite(filename,imgFrame);
            i++;
        }

        // Terminating the webcam if 'q' is pressed
        if(key =='q'){
            std::cout<<"Terminating camera feed!"<<std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}