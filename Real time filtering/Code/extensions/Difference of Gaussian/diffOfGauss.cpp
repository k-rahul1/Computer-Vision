#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>

using std::string;
using namespace std;

//applying gaussian blur with small sigma
int blur3x3( cv::Mat &src, cv::Mat &dst ){
    int row = src.rows;
    int col = src.cols;
    dst = cv::Mat::zeros(row,col,CV_8UC3);
    cv::Mat temp = cv::Mat::zeros(row,col,CV_8UC3);
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = temp.ptr<cv::Vec3b>(i);
        for(int j=1;j<col-1;j++){
            for(int c=0;c<3;c++){
                dp[j][c] = (1*sp[j-1][c]+2*sp[j][c]+1*sp[j+1][c])/4;
            }
        }     
    }

    for(int i=1;i<row-1;i++){
        cv::Vec3b *s1p = temp.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s2p = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *s3p = temp.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *d1p = dst.ptr<cv::Vec3b>(i);
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = (1*s1p[j][c]+2*s2p[j][c]+1*s3p[j][c])/4;
                }
            }
        }     
    return 0;
}

//applying gaussian blur with large sigma
int blur5x5( cv::Mat &src, cv::Mat &dst ){
    int row = src.rows;
    int col = src.cols;
    dst = cv::Mat::zeros(row,col,CV_8UC3);
    cv::Mat temp = cv::Mat::zeros(row,col,CV_8UC3);
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = temp.ptr<cv::Vec3b>(i);
        for(int j=2;j<col-2;j++){
            for(int c=0;c<3;c++){
                dp[j][c] = (1*sp[j-2][c]+4*sp[j-1][c]+6*sp[j][c]+4*sp[j+1][c]+1*sp[j+2][c])/16;
            }
        }     
    }

    for(int i=2;i<row-2;i++){
        cv::Vec3b *s1p = temp.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *s2p = temp.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s3p = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *s4p = temp.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *s5p = temp.ptr<cv::Vec3b>(i+2);
        cv::Vec3b *d1p = dst.ptr<cv::Vec3b>(i);
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = (1*s1p[j][c]+4*s2p[j][c]+6*s3p[j][c]+4*s4p[j][c]+1*s5p[j][c])/16;
                }
            }
        }     
    return 0;
}

int main(){
    cv::Mat image;
    string filename;
    cout<<"Enter the name of image with extension"<<endl;
    cin>>filename;
    image = cv::imread(filename);
    cv::Mat ssigmablur,lsigmablur,diffofG;
    blur3x3(image,ssigmablur); //blur with small sigma
    blur5x5(image,lsigmablur); //blur with large sigma
    diffofG = cv::Mat::zeros(image.size(),CV_8UC3);
    for(int i=0; i<image.rows;i++){
        cv::Vec3b *sp1 = ssigmablur.ptr<cv::Vec3b>(i);
        cv::Vec3b *sp2 = lsigmablur.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = diffofG.ptr<cv::Vec3b>(i);
        for(int j=0; j<image.cols;j++){
            for(int c=0;c<3;c++){
                dp[j][c] = sp1[j][c]-sp2[j][c];  // calculating the difference of two blurred images
            }
        }
    }
    cv::imshow("Difference of Gaussian image", diffofG);
    cv::imwrite("diffOfG.png",diffofG);
    cv::waitKey(0);
    return 0;
}