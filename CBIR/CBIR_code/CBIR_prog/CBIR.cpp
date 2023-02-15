//Created by Rahul Kumar

/*
this program implements the query based CBIR
An query image is input by user and the method to generate the feature vector
the program returns the best N no. of matches defined by user 
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "csv_util.h"
#include <typeinfo>
#include <numeric>
#include "filter.h"
#include "feature_generator.h"
#include "distanceMetric.h"

using std::vector;

int main(){
    char Imgname[256];

    //taking the target image from the user
    std::cout<<"Enter target image name with extension"<<std::endl;
    std::cin>>Imgname;
    cv::Mat Targetimg = cv::imread(Imgname);

    //displaying the options for feature generation
    std::cout<<"Enter the method option for generating feature vector:"<<std::endl;
    std::cout<<"b : baseline"<<std::endl;
    std::cout<<"h : histogram"<<std::endl;
    std::cout<<"m : Multihistogram"<<std::endl;
    std::cout<<"t : colorTexture histogram"<<std::endl;
    std::cout<<"c : custom histogram"<<std::endl;
    std::cout<<"g : gabor histogram"<<std::endl;
    std::cout<<"s : spatial variance"<<std::endl;
    char method[256];
    std::cin>>method;

    //user input for number of matches to be generated
    std::cout<<"Enter the number of matches required"<<std::endl;
    int n;
    std::cin>>n;

    //Baseline feature method is implemented for feature generation
    //and Euclidean distance as distance metric
    if(!strcmp(method,"b")){
      vector<float> TfeatureVec;
      midSquare9x9(Targetimg,TfeatureVec);

      char featureFile[256] = "feature_baseline.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data;

      //reading image name and feature vector from already saved csv file
      read_image_data_csv( featureFile, filenames, data, 0 );

      std::vector<float> distance;
      for(int i=0;i<(int)data.size();i++){
        distance.push_back(blDistanceMetric(TfeatureVec,data[i])); //calculating the distance
      }

      //generating pair of distance and corresponding image
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i])); 
      }
      sort(DistImgPair.begin(),DistImgPair.end()); //sorting as per distance

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;

      //generating best N matches
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }
    }

    //Histogram feature method is implemented for feature generation
    //and Histogram Intersection as distance metric
    else if(!strcmp(method,"h")){
      vector<float> TfeatureVec1;
      Histogram(Targetimg,TfeatureVec1);
      char featureFile[256] = "feature_histogram.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data;
      read_image_data_csv( featureFile, filenames, data, 0 );
      
      std::vector<float> distance;
      for(int i=0;i<(int)data.size();i++){
        distance.push_back(histDistanceMetric(TfeatureVec1,data[i]));
      }
    
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i]));
      }
      sort(DistImgPair.begin(),DistImgPair.end());

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }
    }

    //Multi Histogram feature method is implemented for feature generation
    //and histogram intersection as distance metric
    else if(!strcmp(method,"m")){
      vector<float> TfeatureVec1;
      vector<float> TfeatureVec2;
      MultiHistogram(Targetimg,TfeatureVec1,TfeatureVec2);
      char topfeatureFile[256] = "top_feature_histogram.csv";
      char bottomfeatureFile[256] = "bottom_feature_histogram.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data1;
      std::vector<std::vector<float>> data2;
      read_image_data_csv( topfeatureFile, filenames, data1, 0 );
      read_image_data_csv( bottomfeatureFile, filenames, data2, 0 );
      
      std::vector<float> distance;
      for(int i=0;i<(int)data1.size();i++){
        distance.push_back(MultihistDistanceMetric(TfeatureVec1, TfeatureVec2, data1[i], data2[i]));
      }
    
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data1.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i]));
      }
      sort(DistImgPair.begin(),DistImgPair.end());

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }
    }

    //Color Texture feature method is implemented for feature generation
    //and Euclidean distance as distance metric
    else if(!strcmp(method,"t")){
      vector<float> colorTexturefeatureVec = colorTexture(Targetimg);
      char colorTexturefeatureFile[256] = "colorTexture_feature_histogram.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data;
      read_image_data_csv( colorTexturefeatureFile, filenames, data, 0 );

      std::vector<float> distance;
      for(int i=0;i<(int)data.size();i++){
        distance.push_back(colorTextureDistanceMetric(colorTexturefeatureVec, data[i]));
      }
    
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i]));
      }
      sort(DistImgPair.begin(),DistImgPair.end());

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }
    }

    //Custom feature method is implemented for feature generation
    //and weighted Euclidean distance as distance metric
    else if(!strcmp(method,"c")){
      vector<float> customfeatureVec = custom(Targetimg);
      char customfeatureFile[256] = "custom_histogram.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data;

      read_image_data_csv( customfeatureFile, filenames, data, 0 );
      
      std::vector<float> distance;
      for(int i=0;i<(int)data.size();i++){
        distance.push_back(customDistanceMetric(customfeatureVec,data[i]));
      }
    
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i]));
      }
      sort(DistImgPair.begin(),DistImgPair.end());

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }
    }

    //Gabor feature method is implemented for feature generation
    //and Manhattan Distance as distance metric
    else if(!strcmp(method,"g")){
      vector<float> gaborfeatureVec = gaborFeature(Targetimg);
      char gaborfeatureFile[256] = "gabor_histogram.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data;

      read_image_data_csv( gaborfeatureFile, filenames, data, 0 );
      
      std::vector<float> distance;
      for(int i=0;i<(int)data.size();i++){
        distance.push_back(gaborDistanceMetric(gaborfeatureVec,data[i]));
      }
    
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i]));
      }
      sort(DistImgPair.begin(),DistImgPair.end());

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;
      char imgpath[256];
      strcpy(imgpath, "home/rahul/Desktop/Computer_Vision/assignment2/olympus");
      strcat(imgpath, "/");
      
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }
    }

    //Spatial Variance, canny histogram and color histogram method is implemented for feature generation
    //and Manhattan distance for Colorans texture Histogram and calculating standard deviation for Spatial variance
    else if(!strcmp(method,"s")){
      vector<float> spatialfeatureVec = spatialFeature(Targetimg);
      char spatialfeatureFile[256] = "spatial_histogram.csv";
      std::vector<char *> filenames;
      std::vector<std::vector<float>> data;

      read_image_data_csv( spatialfeatureFile, filenames, data, 0 );

      std::vector<float> distance;
      for(int i=0;i<(int)data.size();i++){
        distance.push_back(spatialDistanceMetric(spatialfeatureVec,data[i]));
      }
    
      std::vector<std::pair<float,char *>> DistImgPair;
      for(int i=0;i<(int)data.size();i++){
        DistImgPair.push_back(std::make_pair(distance[i],filenames[i]));
      }
      sort(DistImgPair.begin(),DistImgPair.end());

      std::cout<<"Best "<< n<< " matches are:"<<std::endl;
      for(int i=1; i<=n;i++){
        std::cout<<DistImgPair[i].second<<std::endl;
      }

    }
    return 0;
}