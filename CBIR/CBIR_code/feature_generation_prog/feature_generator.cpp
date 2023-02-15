//Created by Rahul Kumar

// This program generates various feature vector for CBIR

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <numeric>
#include "filter.h"
#include "feature_generator.h"
#include <cmath>
using std::vector;

//Calculate feature vector with 9x9 square in the middle of the image
int midSquare9x9(cv::Mat &testImage, std::vector<float> &featureVec1){
    int tlx = (testImage.rows/2)-5;
    int tly = (testImage.cols/2)-5;
    for(int i=0;i<9;i++){
        for(int j=0;j<9;j++){
            for(int c=0;c<3;c++){
                featureVec1.push_back(testImage.at<cv::Vec3b>(tlx+i,tly+j)[c]);
            }
        }
    }
    return 0;
}

//Generates single normalized 3D color histogram
int Histogram(cv::Mat &testImage, std::vector<float> &featureVec1){
    int size[3] = {8,8,8};
    cv::Mat hist3d(3,size,CV_32S,cv::Scalar(0)); //3D histogram is declared with 8 bins on each axis
    for(int i=0;i<testImage.rows;i++){
    cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=0; j<testImage.cols;j++){
          int x = (sp[j][0]*8)/256;
          int y = (sp[j][1]*8)/256;
          int z = (sp[j][2]*8)/256;
          hist3d.at<int>(x,y,z) += 1;  //incrementing the bin value 
        }
    }
    cv::Mat flatHist = hist3d.reshape(1,1);
    flatHist.copyTo(featureVec1);
    float imgSz = testImage.rows*testImage.cols;
    for(int i=0; i<(int)featureVec1.size();i++){
        featureVec1[i] = featureVec1[i]/imgSz;    //Normalizing histogram
    }
    return 0;
}

//Generates two(top half and bottom half of image) normalized 3D color histogram
int MultiHistogram(cv::Mat &testImage, std::vector<float> &featureVec1, std::vector<float> &featureVec2){
    int size[3] = {8,8,8};
    cv::Mat hist3d1(3,size,CV_32S,cv::Scalar(0)); // Top 3D histogram is declared with 8 bins on each axis
    cv::Mat hist3d2(3,size,CV_32S,cv::Scalar(0)); // Top 3D histogram is declared with 8 bins on each axis
    int centerRow = testImage.rows/2;
    for(int i=0;i<centerRow;i++){
    cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=0; j<testImage.cols;j++){
            int x = (sp[j][0]*8)/256;
            int y = (sp[j][1]*8)/256;
            int z = (sp[j][2]*8)/256;
            hist3d1.at<int>(x,y,z) += 1;
        }
    }
    for(int i=centerRow;i<testImage.rows;i++){
    cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=0; j<testImage.cols;j++){
          int x = (sp[j][0]*8)/256;
          int y = (sp[j][1]*8)/256;
          int z = (sp[j][2]*8)/256;
          hist3d2.at<int>(x,y,z) += 1;
        }
    }
    cv::Mat flatHist1 = hist3d1.reshape(1,1);
    cv::Mat flatHist2 = hist3d2.reshape(1,1);
    flatHist1.copyTo(featureVec1);
    flatHist2.copyTo(featureVec2);
    float imgSz1 = accumulate(featureVec1.begin(),featureVec1.end(),0);
    float imgSz2 = accumulate(featureVec2.begin(),featureVec2.end(),0);
    for(int i=0; i<(int)featureVec1.size();i++){
        featureVec1[i] = featureVec1[i]/imgSz1;   //Normalizing histogram
    }
    for(int i=0; i<(int)featureVec2.size();i++){
        featureVec2[i] = featureVec2[i]/imgSz2;   //Normalizing histogram
    }
    return 0;
}

//Generates whole image color histogram and a whole image texture histogram
std::vector<float> colorTexture(cv::Mat &testImage){
    int size[3] = {8,8,8};
    cv::Mat hist3d(3,size,CV_32S,cv::Scalar(0));
    for(int i=0;i<testImage.rows;i++){
    cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=0; j<testImage.cols;j++){
          int x = (sp[j][0]*8)/256;
          int y = (sp[j][1]*8)/256;
          int z = (sp[j][2]*8)/256;
          hist3d.at<int>(x,y,z) += 1;
        }
    }
    cv::Mat flatHist = hist3d.reshape(1,1);
    vector<float> colorfeatureVec;
    flatHist.copyTo(colorfeatureVec);
    float imgSz = testImage.rows*testImage.cols;
    for(int i=0; i<(int)colorfeatureVec.size();i++){
        colorfeatureVec[i] = colorfeatureVec[i]/imgSz;
    }
    cv::Mat sobelxImage, sobelyImage, gradMagImage;
    sobelY3x3(testImage,sobelyImage);
    sobelX3x3(testImage,sobelxImage);
    magnitude(sobelxImage,sobelyImage, gradMagImage); //sobel magnitude is calculated to generate texture histogram

    cv::Mat texturehist = cv::Mat::zeros(1,256,CV_32SC1);
    for(int i=0;i<testImage.rows;i++){
    cv::Vec3b *sp = gradMagImage.ptr<cv::Vec3b>(i);
        for(int j=0; j<testImage.cols;j++){
          int maxPixelval = std::max(sp[j][0],std::max(sp[j][1],sp[j][2]));
          texturehist.at<int>(0,maxPixelval) += 1;
        }
    }
    cv::Mat flatTexturehist = texturehist.reshape(1,1);
    vector<float> texturefeatureVec;
    flatTexturehist.copyTo(texturefeatureVec);
    for(int i=0; i<(int)texturefeatureVec.size();i++){
        texturefeatureVec[i] = texturefeatureVec[i]/imgSz;
    }
    vector<float> colorTexturefeatureVec(colorfeatureVec);
    colorTexturefeatureVec.insert(colorTexturefeatureVec.end(), texturefeatureVec.begin(), texturefeatureVec.end());

    return colorTexturefeatureVec;

}

//Generates custom feature vector with 5 - 3D color histogram and 1D texture hostogram
std::vector<float> custom(cv::Mat &testImage){
    int a = 8;
    int b = 8;
    int c = 8;
    int size[3] = {a,b,c};

    //5 color histograms are declared
    cv::Mat hist3d1(3,size,CV_32S,cv::Scalar(0)); 
    cv::Mat hist3d2(3,size,CV_32S,cv::Scalar(0));
    cv::Mat hist3d3(3,size,CV_32S,cv::Scalar(0));
    cv::Mat hist3d4(3,size,CV_32S,cv::Scalar(0));
    cv::Mat hist3d5(3,size,CV_32S,cv::Scalar(0));

    int row = testImage.rows;
    int col = testImage.cols;
    int quatcol = 0.25*col;
    int thirdquatcol = 0.75*col;
    int quatrow = 0.25*row;
    int thirdquatrow = 0.75*row;

    //histogram for left part of image
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=0; j<quatcol;j++){
          int x = (sp[j][0]*a)/256;
          int y = (sp[j][1]*b)/256;
          int z = (sp[j][2]*c)/256;
          hist3d1.at<int>(x,y,z) += 1;
        }
    }

    //histogram for top mid part of image
    for(int i=0;i<quatrow;i++){
        cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=quatcol; j<thirdquatcol;j++){
          int x = (sp[j][0]*a)/256;
          int y = (sp[j][1]*b)/256;
          int z = (sp[j][2]*c)/256;
          hist3d2.at<int>(x,y,z) += 1;
        }
    }

    //histogram for central part of image
    for(int i=quatrow;i<thirdquatrow;i++){
        cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=quatcol; j<thirdquatcol;j++){
          int x = (sp[j][0]*a)/256;
          int y = (sp[j][1]*b)/256;
          int z = (sp[j][2]*c)/256;
          hist3d3.at<int>(x,y,z) += 1;
        }
    }

    //histogram for central part of image
    for(int i=thirdquatrow;i<row;i++){
        cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=quatcol; j<thirdquatcol;j++){
          int x = (sp[j][0]*a)/256;
          int y = (sp[j][1]*b)/256;
          int z = (sp[j][2]*c)/256;
          hist3d4.at<int>(x,y,z) += 1;
        }
    }

    ////histogram for right part of image
    for(int i=0;i<row;i++){
        cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=thirdquatcol; j<col;j++){
          int x = (sp[j][0]*a)/256;
          int y = (sp[j][1]*b)/256;
          int z = (sp[j][2]*c)/256;
          hist3d5.at<int>(x,y,z) += 1;
        }
    }
    
    //canny edge detector is implemented
    cv::Mat gray,blur,canny;
    cv::cvtColor(testImage,gray,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray,blur,cv::Size(5,5),1,1);
    cv::Canny(blur,canny,50,200);

    //1D histogram is formed from canny image
    vector<float> cannyVec(256);
    for(int i=0;i<testImage.rows;i++){
        uchar *sp = canny.ptr<uchar>(i);
        for(int j=0; j<testImage.cols;j++){
          int x = sp[j];
          cannyVec[x]++;
        }
    }

    float imgSz = testImage.rows*testImage.cols;
    for(int i=0; i<(int)cannyVec.size();i++){
        cannyVec[i] = cannyVec[i]/imgSz;
    }


    //5 different vectors are declared to store the histogram data
    cv::Mat flatHist1 = hist3d1.reshape(1,1);
    cv::Mat flatHist2 = hist3d2.reshape(1,1);
    cv::Mat flatHist3 = hist3d3.reshape(1,1);
    cv::Mat flatHist4 = hist3d4.reshape(1,1);
    cv::Mat flatHist5 = hist3d5.reshape(1,1);
    vector<float> featureVec1, featureVec2,featureVec3,featureVec4,featureVec5;
    flatHist1.copyTo(featureVec1);
    flatHist2.copyTo(featureVec2);
    flatHist3.copyTo(featureVec3);
    flatHist4.copyTo(featureVec4);
    flatHist5.copyTo(featureVec5);
    float imgSz1 = accumulate(featureVec1.begin(),featureVec1.end(),0);
    float imgSz2 = accumulate(featureVec2.begin(),featureVec2.end(),0);
    float imgSz3 = accumulate(featureVec3.begin(),featureVec3.end(),0);
    float imgSz4 = accumulate(featureVec4.begin(),featureVec4.end(),0);
    float imgSz5 = accumulate(featureVec5.begin(),featureVec5.end(),0);

    //Normalizing the vecotrs
    for(int i=0; i<(int)featureVec1.size();i++){
        featureVec1[i] = featureVec1[i]/imgSz1;
    }
    for(int i=0; i<(int)featureVec2.size();i++){
        featureVec2[i] = featureVec2[i]/imgSz2;
    }
    for(int i=0; i<(int)featureVec3.size();i++){
        featureVec3[i] = featureVec3[i]/imgSz3;
    }
    for(int i=0; i<(int)featureVec4.size();i++){
        featureVec4[i] = featureVec4[i]/imgSz4;
    }
    for(int i=0; i<(int)featureVec5.size();i++){
        featureVec5[i] = featureVec5[i]/imgSz5;
    }

    //single vector is generated
    vector<float> customfeatureVec(featureVec1);
    customfeatureVec.insert(customfeatureVec.end(), featureVec2.begin(), featureVec2.end());
    customfeatureVec.insert(customfeatureVec.end(), featureVec3.begin(), featureVec3.end());
    customfeatureVec.insert(customfeatureVec.end(), featureVec4.begin(), featureVec4.end());
    customfeatureVec.insert(customfeatureVec.end(), featureVec5.begin(), featureVec5.end());
    customfeatureVec.insert(customfeatureVec.end(), cannyVec.begin(), cannyVec.end());

    return customfeatureVec;
}

//Generates gabor histograms with 6 different settings
std::vector<float> gaborFeature(cv::Mat &testImage){
    cv::Mat GF1,GF2,GF3,GF4,GF5,GF6;
    double lambda = 0.5;
    double gama = 0.5;
    double psy = 0;
    cv::cvtColor(testImage,testImage,cv::COLOR_BGR2GRAY);

    //1st gabor filter is implemented with theta pi/6, sigma 2, lambda 0.5
    cv::Mat kernal1 = cv::getGaborKernel(cv::Size(5,5),2, M_PI/6, lambda, gama, psy, CV_32F);
    cv::filter2D(testImage,GF1,CV_32F, kernal1);
    GF1.convertTo(GF1, CV_8UC1);

    //2nd gabor filter is implemented with theta pi/3, sigma 2.5, lambda 0.5/2
    cv::Mat kernal2 = cv::getGaborKernel(cv::Size(11,11),2.5, M_PI/3, lambda/2, gama, psy, CV_32F);
    cv::filter2D(testImage,GF2,CV_32F, kernal2);
    GF2.convertTo(GF2, CV_8UC1);

    //3rd gabor filter is implemented with theta pi/2, sigma 3, lambda 0.5/4
    cv::Mat kernal3 = cv::getGaborKernel(cv::Size(17,17),3, M_PI/2, lambda/4, gama, psy, CV_32F);
    cv::filter2D(testImage,GF3,CV_32F, kernal3);
    GF3.convertTo(GF3, CV_8UC1);

    //4th gabor filter is implemented with theta pi*2/3, sigma 3.5, lambda 0.5/8
    cv::Mat kernal4 = cv::getGaborKernel(cv::Size(23,23),3.5, M_PI*2/3, lambda/8, gama, psy, CV_32F);
    cv::filter2D(testImage,GF4,CV_32F, kernal4);
    GF4.convertTo(GF4, CV_8UC1);

    //5th gabor filter is implemented with theta pi*5/6, sigma 4, lambda 0.5/16
    cv::Mat kernal5 = cv::getGaborKernel(cv::Size(29,29),4, M_PI*5/6, lambda/16, gama, psy, CV_32F);
    cv::filter2D(testImage,GF5,CV_32F, kernal5);
    GF5.convertTo(GF5, CV_8UC1);

    //6th gabor filter is implemented with theta pi, sigma 4.5, lambda 0.5/32
    cv::Mat kernal6 = cv::getGaborKernel(cv::Size(35,35),4.5, 0, lambda/32, gama, psy, CV_32F);
    cv::filter2D(testImage,GF6,CV_32F, kernal6);
    GF6.convertTo(GF6, CV_8UC1);

    cv::Mat gaborImage = cv::Mat::zeros(GF1.size(), CV_8UC1);
    vector<float> gaborVec(256);
    for(int i=0;i<testImage.rows;i++){
        uchar *sp1 = GF1.ptr<uchar>(i);
        uchar *sp2 = GF2.ptr<uchar>(i);
        uchar *sp3 = GF3.ptr<uchar>(i);
        uchar *sp4 = GF4.ptr<uchar>(i);
        uchar *sp5 = GF5.ptr<uchar>(i);
        uchar *sp6 = GF6.ptr<uchar>(i);
        uchar *dp = gaborImage.ptr<uchar>(i);

        //Combining all gabor filter results into one image
        for(int j=0; j<testImage.cols;j++){
          dp[j] = (pow((sp1[j]*sp1[j]+sp2[j]*sp2[j]+sp3[j]*sp3[j]+sp4[j]*sp4[j]+sp5[j]*sp5[j]+sp6[j]*sp6[j]),0.5))/pow(6,0.5);
          int x = dp[j];
          gaborVec[x]++;
        }
    }

    //Normalizing the histograms
    float imgSz = testImage.rows*testImage.cols;
    for(int i=0; i<(int)gaborVec.size();i++){
        gaborVec[i] = gaborVec[i]/imgSz;
    }

    return gaborVec;
}

//Generates spatial variance feature along with color and texture histogram in HSV color space to detect banana
std::vector<float> spatialFeature(cv::Mat &testImage){
    cv::Mat testImage1;
    testImage.copyTo(testImage1);
    cv::cvtColor(testImage,testImage,cv::COLOR_BGR2HSV);
    int size[3] = {8,16,4};
    cv::Mat hist3d(3,size,CV_32S,cv::Scalar(0));
    vector<float> xpixelPos;
    vector<float> ypixelPos;

    //setting the range to detect yellow color of banana
    int hmin = 20, smin =160, vmin = 128;
    int hmax = 45, smax = 255, vmax =255;

    //Generating the color and pixel position vectors for central part of image
    for(int i=testImage.rows*0.25;i<testImage.rows*0.75;i++){
        cv::Vec3b *sp = testImage.ptr<cv::Vec3b>(i);
        for(int j=testImage.cols*0.25; j<testImage.cols*0.75;j++){
            //pixels present in the defined range only used to create feature vector
            if(sp[j][0]>=hmin && sp[j][0]<=hmax && sp[j][1]>=smin && sp[j][1]<=smax && sp[j][2]>=vmin && sp[j][2]<=vmax){
                int x = (sp[j][0]*8)/181;
                int y = (sp[j][1]*16)/256;
                int z = (sp[j][2]*4)/256;
                hist3d.at<int>(x,y,z) += 1;
                xpixelPos.push_back(j);     //yellow pixel X coordinate 
                ypixelPos.push_back(i);     //yellow pixel y coordinate 
            }
        }
    }

    cv::Mat flatHist = hist3d.reshape(1,1);
    vector<float> colorfeatureVec;
    flatHist.copyTo(colorfeatureVec);
    float imgsize = testImage.rows*0.5*testImage.cols*0.5;
    for(int i=0; i<(int)colorfeatureVec.size();i++){
        colorfeatureVec[i] = colorfeatureVec[i]/imgsize;
    }
    vector<float> pixelPos(colorfeatureVec);

    //Implementing canny edge detector
    cv::Mat gray,blur,canny;
    cv::cvtColor(testImage1,gray,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray,blur,cv::Size(5,5),1,1);
    cv::Canny(blur,canny,50,300);

    //1D histogram is generated using canny image
    vector<float> cannyVec(256);
    for(int i=testImage.rows*0.25;i<testImage.rows*0.75;i++){
        uchar *sp = canny.ptr<uchar>(i);
        for(int j=testImage.cols*0.25; j<testImage.cols*0.75;j++){
          int x = sp[j];
          cannyVec[x]++;
        }
    }

    float imgSz = testImage.rows*0.5*testImage.cols*0.5;
    for(int i=0; i<(int)cannyVec.size();i++){
        cannyVec[i] = cannyVec[i]/imgSz;
    }

    pixelPos.insert(pixelPos.end(),cannyVec.begin(),cannyVec.end());

    int n = xpixelPos.size();
    float varX = 0, varY = 0;

    //setting high variance for images with very less no. of yellow pixels
    if(n<4500){
        varX = 100000;
        varY = 100000;
    }
    else{
        float meanX = (accumulate(xpixelPos.begin(),xpixelPos.end(),0))/n; //Calculating mean of position X
        float meanY = (accumulate(ypixelPos.begin(),ypixelPos.end(),0))/n; //Calculating mean of position Y

        //Calculating the variance for X and Y coordinates
        for(int i=0;i<n;i++){
            varX += pow((xpixelPos[i]-meanX),2); 
            varY += pow((ypixelPos[i]-meanY),2);
        }
        varX = varX/n;
        varY = varY/n;
    }
    pixelPos.push_back(varX);
    pixelPos.push_back(varY);
    pixelPos.push_back(n);
    return pixelPos;
}