//Created by Rahul Kumar

//it implements different distance metric on feature vector

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "distanceMetric.h"
#include "csv_util.h"
#include <cmath>

//Implementing Euclidean distance as distance metric
float blDistanceMetric(std::vector<float> &target, std::vector<float> &feature){
    float res = 0;
    std::vector<float> distance;
    for(int i=0; i<(int)target.size();i++){
        res = res + pow((target[i]-feature[i]),2);  //calculating Euclidean distance
    }
    return res;
}

//Implementing Histogram Intersection as distance metric
float histDistanceMetric(std::vector<float> &target, std::vector<float> &feature){
    float res = 0;
    for(int i=0; i<(int)target.size();i++){
        res = res + std::min(target[i],feature[i]); //Calculating intersection by choosing the min value
    }
    return (1-res);
}

//Implementing histogram intersection as distance metric
float MultihistDistanceMetric(std::vector<float> &target1, std::vector<float> &target2, std::vector<float> &feature1, std::vector<float> &feature2){
    float res1= 0;
    float res2= 0;
    for(int i=0; i<(int)target1.size();i++){
        res1 = res1 + std::min(target1[i],feature1[i]);
        res2 = res2 + std::min(target2[i],feature2[i]);
    }
    return (1-(0.5*(res1+res2))); //applying same weight to both top and bottom features
}

//Implementing Euclidean distance as distance metric
float colorTextureDistanceMetric(std::vector<float> &target, std::vector<float> &feature){
    float res = 0;
    for(int i=0; i<(int)target.size();i++){
        res = res + pow((target[i]-feature[i]),2);
    }
    return res;
}

//Implementing weighted Euclidean distance as distance metric
float customDistanceMetric(std::vector<float> &target, std::vector<float> &feature){
    float res = 0,res1 =0;
    
    //applying more weight to central part than the outer parts of image
    for(int i=0; i<512;i++){
        res = res + 0.1*(pow((target[i]-feature[i]),2))
        +0.2*(pow((target[512+i]-feature[512+i]),2))
        +0.4*(pow((target[512*2+i]-feature[512*2+i]),2))
        +0.2*(pow((target[512*3+i]-feature[512*3+i]),2))
        +0.1*(pow((target[512*4+i]-feature[512*4+i]),2));
    }
    for(int i=2560;i<2816;i++){
        res1 =res1 + pow((target[i]-feature[i]),2);
    }
    return (0.7*res+0.3*res1);
}

// Implementing Manhattan Distance as distance metric
float gaborDistanceMetric(std::vector<float> &target, std::vector<float> &feature){
    float res = 0;
    for(int i=0; i<(int)target.size();i++){
        res = res + abs(target[i]-feature[i]);
    }
    return res;
}

//Implementing Manhattan distance for Color and Canny Histogram and calculating standard deviation for Spatial variance
float spatialDistanceMetric(std::vector<float> &target, std::vector<float> &feature){
    float res, xVar, yVar,posVar, colorVar = 0, cannyVar;
    for(int i=0; i<512;i++){
        colorVar = colorVar + abs(target[i]-feature[i]);
    }

    for(int i=512; i<768;i++){
        cannyVar = cannyVar + abs(target[i]-feature[i]);
    }

    xVar = feature[768]/target[768];
    yVar = feature[769]/target[769];
    posVar = abs(2-(xVar + yVar));
    float posDeviation = pow(posVar,0.5);
    res = 0.3*colorVar + 0.5*posDeviation + 0.2*cannyVar;
    return res;
}