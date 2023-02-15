/*
  Bruce A. Maxwell
  S21
  
  Sample code to identify image fils in a directory

  //Modified by//
  Rahul Kumar 

  After reading the image, code will generate feature vectors and create a csv file containing
  image name and feature vector 
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include "csv_util.h"
#include <vector>
#include <numeric>
#include "filter.h"
#include "feature_generator.h"

using std::vector;

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  DIR *dirp;
  struct dirent *dp;

  // check for sufficient arguments
  if( argc < 2) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  //Filenames of csv are declared
  char bfeatureFile[256] = "feature_baseline.csv";
  char hfeatureFile[256] = "feature_histogram.csv";
  char htopfeatureFile[256] = "top_feature_histogram.csv";
  char hbottomfeatureFile[256] = "bottom_feature_histogram.csv";
  char colorTexturefeatureFile[256] = "colorTexture_feature_histogram.csv";
  char customfeatureFile[256] = "custom_histogram.csv";
  char gaborfeatureFile[256] = "gabor_histogram.csv";
  char spatialfeatureFile[256] = "spatial_histogram.csv";

  //User input is taken for method to generate feature vector
  std::cout<<"Enter the method option for generating feature vector"<<std::endl;
  std::cout<<"b: baseline"<<std::endl;
  std::cout<<"h: histogram"<<std::endl;
  std::cout<<"m: Multihistogram"<<std::endl;
  std::cout<<"t: colorTexture histogram"<<std::endl;
  std::cout<<"c: custom"<<std::endl;
  std::cout<<"g: gabor histogram"<<std::endl;
  std::cout<<"s: spatial variance"<<std::endl;
  char method[256];
  std::cin>>method;

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	    strstr(dp->d_name, ".png") ||
	    strstr(dp->d_name, ".ppm") ||
	    strstr(dp->d_name, ".tif") ) {

      //printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      //printf("full path name: %s\n", buffer);
    }
    else{
      continue;
    }

    //Baseline feature method is implemented
    if(!strcmp(method,"b")){
      cv::Mat testImage = cv::imread(buffer);
      vector<float> featureVec;
      midSquare9x9(testImage, featureVec); // function call for baseline feature method
      append_image_data_csv( bfeatureFile, dp->d_name, featureVec, 0 ); //Appending the image name and feature vector to csv file
    }
    //Histogram feature method is implemented
    else if(!strcmp(method,"h")){
      cv::Mat testImage = cv::imread(buffer);
      vector<float> featureVec;
      Histogram(testImage,featureVec); // function call for baseline feature method
      append_image_data_csv( hfeatureFile, dp->d_name, featureVec, 0 );
    }
    //Multi Histogram feature method is implemented
    else if(!strcmp(method,"m")){
      cv::Mat testImage = cv::imread(buffer);
      vector<float> featureVec1;
      vector<float> featureVec2;
      MultiHistogram(testImage,featureVec1,featureVec2); // function call for Multihistogram feature method
      append_image_data_csv( htopfeatureFile, dp->d_name, featureVec1, 0 ); //csv file for top histogram
      append_image_data_csv( hbottomfeatureFile, dp->d_name, featureVec2, 0 ); //csv file for bottom histogram
    }
    //Color Texture feature method is implemented
    else if(!strcmp(method,"t")){
      cv::Mat testImage = cv::imread(buffer);     
      vector<float> colorTexturefeatureVec = colorTexture(testImage); // function call for color texture feature method
      append_image_data_csv( colorTexturefeatureFile, dp->d_name, colorTexturefeatureVec, 0 );
    }
    //Custom feature method is implemented
    else if(!strcmp(method,"c")){
      cv::Mat testImage = cv::imread(buffer);
      vector<float> customfeatureVec = custom(testImage); // function call for custom feature method
      append_image_data_csv( customfeatureFile, dp->d_name, customfeatureVec, 0 );

    }
    //Gabor feature method is implemented
    else if(!strcmp(method,"g")){
      cv::Mat testImage = cv::imread(buffer);
      vector<float> gaborfeatureVec = gaborFeature(testImage); // function call for gabor feature method
      append_image_data_csv( gaborfeatureFile, dp->d_name, gaborfeatureVec, 0 );
    }
    //Spatial Variance feature method is implemented
    else if(!strcmp(method,"s")){
      cv::Mat testImage = cv::imread(buffer);
      vector<float> spatialfeatureVec = spatialFeature(testImage); // function call for spatial feature method
      append_image_data_csv( spatialfeatureFile, dp->d_name, spatialfeatureVec, 0 );
    }
  }
  return(0);
}


