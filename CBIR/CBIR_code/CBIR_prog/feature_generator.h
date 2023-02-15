//Created by Rahul Kumar
//Function declation for different feature generation method

//Calculate feature vector with 9x9 square in the middle of the image
int midSquare9x9(cv::Mat &testImage, std::vector<float> &featureVec1);

//Generates single normalized 3D color histogram
int Histogram(cv::Mat &testImage, std::vector<float> &featureVec1);

//Generates two(top half and bottom half of image) normalized 3D color histogram
int MultiHistogram(cv::Mat &testImage, std::vector<float> &featureVec1, std::vector<float> &featureVec2);

//Generates whole image color histogram and a whole image texture histogram
std::vector<float> colorTexture(cv::Mat &testImage);

//Generates custom feature vector with 5 - 3D color histogram and 1D texture histogram
std::vector<float> custom(cv::Mat &testImage);

//Generates gabor histograms with 6 different settings
std::vector<float> gaborFeature(cv::Mat &testImage);

//Generates spatial variance feature along with color and texture histogram in HSV color space to detect banana
std::vector<float> spatialFeature(cv::Mat &testImage);