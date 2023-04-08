//Created by Rahul Kumar

//Declation of function which are used to implement different morphological filters for 2-D object recognition

//Function to apply threshold to seperate foreground from background
int thresholding(cv::Mat src, cv::Mat &dst);

//Function to cleanup the thresholded video by filling holes and removing spots
int clean(cv::Mat &src, cv::Mat &dst);

//Function to segment the foreground into contours using connected components
std::vector<int> Segment(cv::Mat &src, cv::Mat &dst);

//Function to generate feature vector of objects
std::vector<float> Feature(cv::Mat &src, cv::Mat &dst, cv::Mat label, int regionId, std::vector<int> &reqRegionId);

//function to calculate the distance of unknown object from the labeled objects using Scaled Euclidean
std::vector<float> distanceMetric(std::vector<float> feature, std::vector<std::vector<float>> &database);