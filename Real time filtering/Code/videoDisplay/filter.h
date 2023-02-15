//alternate method to obtain grayscale image
int greyscale(cv::Mat &src,cv::Mat &dst);

//applying gaussian blur to image
int blur5x5( cv::Mat &src, cv::Mat &dst );

//applying horizontal sobel filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst );

//applying vertical sobel filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

//gradient magnitude image using sobel x and y
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

//applying blur and quantizes a color image
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

//video cartoonisation using gradient magnitude and blur/quantize filter
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

//color negative 
int Negative(cv::Mat &src, cv::Mat &dst);