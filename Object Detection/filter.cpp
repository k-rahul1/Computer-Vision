//Created by Rahul Kumar

//Code to implement different morphological filters for 2-D object recognition

#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include"filter.h"


using std::max;
using std::min;
using std::vector;
using std::string;

//Function to apply threshold to seperate foreground from background
int thresholding(cv::Mat src, cv::Mat &dst){
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::cvtColor(src,src,cv::COLOR_BGR2GRAY); //Converting to grayscale
    cv::GaussianBlur(src,src,cv::Size(5,5),1,1); //blurring the image to remove noise
    vector<float> hist(256);

    //generating histogram to find out the optimal position to threshold
    for(int i=0;i<src.rows;i++){
        uchar *sp = src.ptr<uchar>(i);
        for(int j=0;j<src.cols;j++){
            int x = sp[j];
            hist[x]++;
        }
    }

    //applying threshold to the video
    for(int i=0;i<src.rows;i++){
        uchar *sp = src.ptr<uchar>(i);
        uchar *dp = dst.ptr<uchar>(i);
        for(int j=0;j<src.cols;j++){
            if(i<25){
                dp[j]=0;
            }
            else if(sp[j]>130){
                dp[j]=0;
            }
            else{
                dp[j]=255;
            }
        }
    }

    return 0;
}

//Function to cleanup the thresholded video by filling holes and removing spots
int clean(cv::Mat &src, cv::Mat &dst){
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    //creating structuring element to apply dilation and erosion
    cv::Mat kernald = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::Mat kernale = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5,5));
    cv::Mat erodeImg, dilateImg;
    cv::dilate(src,dilateImg,kernald,cv::Point(-1,-1),3); //dilation to fill holes
    cv::erode(dilateImg,erodeImg,kernald,cv::Point(-1,-1),3); //erosion to retain shape
    dst = erodeImg;
    return 0;
}

//Function to segment the foreground into contours using connected components
vector<int> Segment(cv::Mat &src, cv::Mat &dst){
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    cv::Mat label,stat,centroid;

    //Applying connected component function to get region map and region Id
    int regionId = cv::connectedComponentsWithStats(src,label,stat,centroid,8);

    vector<int> modRegion;

    //disregaring contours of very less and very large area
    for(int i=0;i<regionId;i++){
        if((stat.at<int>(cv::Point(4,i))>1200)&&(stat.at<int>(cv::Point(4,i))<160000)){
            modRegion.push_back(i);
        }
    }

    //generating different colors for different contours
    vector<uchar> b,g,r;
    for(int i=0;i<6;i++){
        b.push_back(255-40*i);
        g.push_back(40*i);
        r.push_back(255-40*i);
    }

    //applying different colors to different regions
    for(int i=0; i<label.rows; i++){
        int *sp = label.ptr<int>(i);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(i);
        for(int j=0; j<label.cols; j++){
            if(std::find(modRegion.begin(), modRegion.end(), sp[j]) != modRegion.end()){
                uchar c = sp[j];
                dp[j][0] = b[c];
                dp[j][1] = g[c];
                dp[j][2] = r[c];
            }
        }
    }

    return modRegion;
}

//Function to generate feature vector of objects
vector<float> Feature(cv::Mat &src, cv::Mat &dst, cv::Mat label, int regionId, std::vector<int> &reqRegionId){
    cv::Mat gray;
    cv::cvtColor( src, gray, cv::COLOR_BGR2GRAY ); //converting to grayscale
    cv::GaussianBlur( gray, gray, cv::Size(5,5),1); //applying blur 
    cv::Mat canny_output;
    cv::Canny( gray, canny_output, 50, 200 ); //applying canny edge detector
    vector<vector<cv::Point> > contour;
    
    //finding the contour points of different regions
    cv::findContours( canny_output, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    dst = src.clone();
    vector<cv::Moments> moment(contour.size()); 
    vector<vector<double>> huMoment(contour.size(),vector<double>(7));
    vector<double> alpha(contour.size());
    vector<cv::RotatedRect> obb(contour.size());

    //calculating the invariant moments and axis of least central moment
    for(int i=0; i<contour.size();i++){
        moment[i] = cv::moments(contour[i],false);
        cv::HuMoments(moment[i],huMoment[i]);
        obb[i] = cv::minAreaRect(contour[i]);
        alpha[i] = moment[i].mu20==moment[i].mu02?0:0.5*atan2((2*moment[i].mu11),(moment[i].mu20-moment[i].mu02));
    }

    
    vector<float> feature;
    for(int i=0; i<contour.size();i++){
        cv::Point2f obb_corner[4];
        obb[i].points(obb_corner);
        for(int j=0; j<4; j++){
            //drawing the oriented boundary box for each contour
            cv::line(dst, obb_corner[j], obb_corner[(j+1)%4], cv::Scalar(0,255,0)); 
        }

        float length = obb[i].size.height>obb[i].size.width?obb[i].size.height:obb[i].size.width;
        cv::Point P1;
        P1.x = moment[i].m10/moment[i].m00;
        P1.y = moment[i].m01/moment[i].m00;
        cv::Point P2;
        P2.x =  (int)round(P1.x + 0.5*length * cos(alpha[i]));
        P2.y =  (int)round(P1.y + 0.5*length * sin(alpha[i]));

        //drawing the least central moment axis for each contour
        cv::line(dst,P1,P2,cv::Scalar(255,0,0));

        vector<float> invarMoment(7);    
        for(int j=0; j<7;j++){
            //scaling the huMoment values
            invarMoment[j] = -1 * copysign(1.0, huMoment[i][j]) * log10(huMoment[i][j]>0?huMoment[i][j]:(-1*huMoment[i][j]));
        }
        string text = std::to_string(invarMoment[0]);

        //displaying the huMoment feature in real time for each contour
        cv::putText(dst,text,P1,cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(255,0,0),1);

        //calculating the %filled of area of contour
        float areaRatio = contourArea(contour[i])/(obb[i].size.height*obb[i].size.width);
        
        //calcuating the height to width ratio
        float dimRatio = std::min(obb[i].size.height,obb[i].size.width)/std::max(obb[i].size.height,obb[i].size.width);

        //generating the feature vector for each contour
        feature.push_back(invarMoment[0]);
        feature.push_back(invarMoment[1]);
        feature.push_back(invarMoment[2]);
        feature.push_back(invarMoment[3]);
        feature.push_back(areaRatio);
        feature.push_back(dimRatio);
    }
    return feature;
}

//function to calculate the distance of unknown object from the labeled objects using Scaled Euclidean
std::vector<float> distanceMetric(std::vector<float> feature, std::vector<std::vector<float>> &database){
    int row = database.size();
    int col = database[0].size();
    vector<float> stdDev(col);
    vector<float> mean(col);

    //calculating the mean of each feature
    for(int j=0; j<col;j++){
        for(int i=0; i<row;i++){
            mean[j] += database[i][j];
        }
        mean[j] = mean[j]/row;
    }

    //calculating the standard deviation of each feature
    for(int j=0; j<col;j++){
        for(int i=0; i<row;i++){
            stdDev[j] = stdDev[j] + (database[i][j]-mean[j])*(database[i][j]-mean[j]);
        }
        stdDev[j] = sqrt(stdDev[j]/row);
    }

    vector<float> distance(row);
    //calculating the scaled Euclidean distance
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            distance[i] += ((feature[j]-database[i][j])/stdDev[j])*((feature[j]-database[i][j])/stdDev[j]);
        }
    }
    return distance;
}

