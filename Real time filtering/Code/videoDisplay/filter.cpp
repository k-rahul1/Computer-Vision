#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include"filter.h"

using std::max;

//alternate method to obtain grayscale image
int greyscale(cv::Mat &src,cv::Mat &dst){
    dst = cv::Mat::zeros(src.rows,src.cols,CV_8UC1);
    for(int i=0; i<src.rows;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        uchar *dp = dst.ptr<uchar>(i);
        for(int j=0; j<src.cols;j++){
            dp[j] = max(sp[j][0],max(sp[j][1],sp[j][2]));
        }
    }
    return 0;
}

//applying gaussian blur to image
int blur5x5( cv::Mat &src, cv::Mat &dst ){
    int row = src.rows;
    int col = src.cols;
    dst = cv::Mat::zeros(row,col,CV_8UC3);
    cv::Mat temp = cv::Mat::zeros(row,col,CV_8UC3);
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = temp.ptr<cv::Vec3b>(i);

        //blurring the boundaries
        for(int c=0;c<3;c++){
            dp[0][c] = (1*sp[1][c]+2*sp[0][c]+4*sp[0][c]+2*sp[1][c]+1*sp[2][c])/10;
            dp[1][c] = (1*sp[0][c]+2*sp[0][c]+4*sp[1][c]+2*sp[2][c]+1*sp[3][c])/10;
            dp[col-1][c] = (1*sp[col-3][c]+2*sp[col-2][c]+4*sp[col-1][c]+2*sp[col-1][c]+1*sp[col-2][c])/10;
            dp[col-2][c] = (1*sp[col-4][c]+2*sp[col-3][c]+4*sp[col-2][c]+2*sp[col-1][c]+1*sp[col-1][c])/10;
        }

        //blurring the rest of pixels
        for(int j=2;j<col-2;j++){
            for(int c=0;c<3;c++){
                dp[j][c] = (1*sp[j-2][c]+2*sp[j-1][c]+4*sp[j][c]+2*sp[j+1][c]+1*sp[j+2][c])/10;
            }
        }     
    }

    for(int i=0;i<row-2;i++){
        cv::Vec3b *s1p = temp.ptr<cv::Vec3b>(i-2);
        cv::Vec3b *s2p = temp.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *s3p = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *s4p = temp.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *s5p = temp.ptr<cv::Vec3b>(i+2);
        cv::Vec3b *d1p = dst.ptr<cv::Vec3b>(i);

        //blurring the boundaries
        if(i==0){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                d1p[j][c] = (1*s4p[j][c]+2*s3p[j][c]+4*s3p[j][c]+2*s4p[j][c]+1*s5p[j][c])/10;
                }
            }
        }
        else if(i==1){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                d1p[j][c] = (1*s2p[j][c]+2*s2p[j][c]+4*s3p[j][c]+2*s4p[j][c]+1*s5p[j][c])/10;
                }
            }
        }

        //blurring the rest of pixels
        else if(i>=2&&i<row-2){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = (1*s1p[j][c]+2*s2p[j][c]+4*s3p[j][c]+2*s4p[j][c]+1*s5p[j][c])/10;
                }
            }
        }     
    }
    return 0;
}

//applying horizontal sobel filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    int row = src.rows;
    int col = src.cols;
    dst = cv::Mat::zeros(src.size(),CV_16SC3);
    cv::Mat temp = cv::Mat::zeros(src.size(),CV_16SC3);
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *dp = temp.ptr<cv::Vec3s>(i);

        //operating on boundary pixels
        for(int c=0;c<3;c++){
            dp[0][c] = -1*sp[0][c]+1*sp[1][c];
            dp[col-1][c] = -1*sp[col-2][c]+1*sp[col-1][c];
        }

        //operating on rest of pixels
        for(int j=1;j<col-1;j++){
            for(int c=0;c<3;c++){
                dp[j][c] = -1*sp[j-1][c]+1*sp[j+1][c];
            }
        }     
    }

    for(int i=0;i<row;i++){
        cv::Vec3s *s1p = temp.ptr<cv::Vec3s>(i-1);
        cv::Vec3s *s2p = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *s3p = temp.ptr<cv::Vec3s>(i+1);
        cv::Vec3s *d1p = dst.ptr<cv::Vec3s>(i);

        //operating on boundary pixels
        if(i==0){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = (1*s2p[j][c]+2*s2p[j][c]+1*s3p[j][c])/4;
                }
            }
        }

        //operating on rest of pixels
        else if(i>=1&&i<row-1){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = (1*s1p[j][c]+2*s2p[j][c]+1*s3p[j][c])/4;
                }
            }
        }

        //operating on boundary pixels
        else{
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = (1*s1p[j][c]+2*s2p[j][c]+1*s2p[j][c])/4;
                }
            }
        }     
    }
    return 0;
}

//applying vertical sobel filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    int row = src.rows;
    int col = src.cols;
    dst = cv::Mat::zeros(src.size(),CV_16SC3);
    cv::Mat temp = cv::Mat::zeros(src.size(),CV_16SC3);
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *dp = temp.ptr<cv::Vec3s>(i);

        //operating on boundary pixels
        for(int c=0;c<3;c++){
            dp[0][c] = (1*sp[0][c]+2*sp[0][c]+1*sp[1][c])/4;
            dp[col-1][c] = (1*sp[col-2][c]+2*sp[col-1][c]+1*sp[col-1][c])/4;
        }

        //operating on rest of pixels
        for(int j=1;j<col-1;j++){
            for(int c=0;c<3;c++){
            dp[j][c] = (1*sp[j-1][c]+2*sp[j][c]+1*sp[j][c])/4;
            }
        }     
    }

    for(int i=0;i<row;i++){
        cv::Vec3s *s1p = temp.ptr<cv::Vec3s>(i-1);
        cv::Vec3s *s2p = temp.ptr<cv::Vec3s>(i);
        cv::Vec3s *s3p = temp.ptr<cv::Vec3s>(i+1);
        cv::Vec3s *d1p = dst.ptr<cv::Vec3s>(i);

        if(i==0){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = -1*s2p[j][c]+1*s3p[j][c];
                }
            }
        }
        else if(i>=1&&i<row-1){
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = -1*s1p[j][c]+1*s3p[j][c];
                }
            }
        }
        else{
            for(int j=0;j<col;j++){
                for(int c=0;c<3;c++){
                    d1p[j][c] = -1*s1p[j][c]+1*s2p[j][c];
                }
            }
        }     
    }
    return 0;
}

//gradient magnitude image using sobel x and y
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    dst = cv::Mat::zeros(sx.size(),CV_8UC3);
    for(int i=0;i<sx.rows;i++){
        cv::Vec3s *sxp = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syp = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);
        for(int j=0; j<sx.cols;j++){
            for(int c=0;c<3;c++){
                dp[j][c] = pow((sxp[j][c]*sxp[j][c]+syp[j][c]*syp[j][c]),0.5);
            }
        }
    }
    return 0;
}

//applying blur and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    int b = 255/levels;
    cv::Mat test;
    blur5x5(src,test); // applying blur
    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    for(int i=0;i<src.rows;i++){
        cv::Vec3b *sp = test.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            dp[j][0] = ((sp[j][0])/b)*b;
            dp[j][1] = ((sp[j][1])/b)*b;
            dp[j][2] = ((sp[j][2])/b)*b;
        }
    }
    return 0;
}

//video cartoonisation using gradient magnitude and blur/quantize filter
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ){
    cv::Mat Sobelx,Sobely,GradMag,BlurQuant;
    sobelX3x3(src,Sobelx);
    sobelY3x3(src,Sobely);
    magnitude(Sobelx, Sobely, GradMag);
    blurQuantize(src,BlurQuant,levels);
    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    for(int i=0; i<src.rows;i++){
        cv::Vec3b *mp = GradMag.ptr<cv::Vec3b>(i);
        cv::Vec3b *bp = BlurQuant.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);
        for(int j=0; j<src.cols;j++){
            for(int c=0;c<3;c++){
                if(mp[j][c]>magThreshold){
                    dp[j][c]=0;
                }
                else{
                    dp[j][c]=bp[j][c];
                }
            }
        }
    }
    return 0;
}

//color negative
int Negative(cv::Mat &src, cv::Mat &dst){
    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    for(int i=0; i<src.rows;i++){
        cv::Vec3b *sp = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dp[j][c]= 255-sp[j][c];
            }
        }
    }
    return 0;
}