#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>

using std::string;
using namespace std;

int main(){
    cv::Mat image;
    string filename;
    cout<<"Enter the name of image with extension"<<endl;
    cin>>filename;
    image = cv::imread(filename);
    cv::Mat medianImg;
    medianImg = cv::Mat::zeros(image.size(),CV_8UC3);
    for(int i=1;i<image.rows-1;i++){
        cv::Vec3b *sp1 = image.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *sp2 = image.ptr<cv::Vec3b>(i);
        cv::Vec3b *sp3 = image.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *dp = medianImg.ptr<cv::Vec3b>(i);
        for(int j=1;j<image.cols-1;j++){
            for(int c=0;c<3;c++){
                    unsigned char array[] = {sp1[j-1][c],sp1[j][c],sp1[j+1][c],
                                             sp2[j-1][c],sp2[j][c],sp2[j+1][c],
                                             sp3[j-1][c],sp3[j][c],sp3[j+1][c]}; // storing the 9 pixel values
                    sort(array,array+9); //sorting the 9 pixel values
                    dp[j][c]=array[4]; // finding the median 
            }
        }
    }
    
    cv::Vec3b *dp = medianImg.ptr<cv::Vec3b>(0);
    cv::Vec3b *sp = medianImg.ptr<cv::Vec3b>(1);
    cv::Vec3b *dp1 = medianImg.ptr<cv::Vec3b>(image.rows-1);
    cv::Vec3b *sp1 = medianImg.ptr<cv::Vec3b>(image.rows-2);

    //for edge cases , boundary rows
    for(int j=1;j<image.cols-1;j++){
        for(int c=0;c<3;c++){
            dp[j][c]=sp[j][c];
            dp1[j][c]=sp1[j][c];
        }
    }
    
    //for edge cases , boundary cols
    for(int i=0;i<image.rows;i++){
        cv::Vec3b *dp = medianImg.ptr<cv::Vec3b>(i);
        for(int c=0;c<3;c++){
            dp[0][c]=dp[1][c];
            dp[image.cols-1][c]=dp[image.cols-2][c];
        }
    }
    
    cv::imshow("Median filter",medianImg);
    cv::imwrite("Median filter.png",medianImg);
    cv::waitKey(0);
    return 0;
}