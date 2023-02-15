#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>

using std::string;
using namespace std;
using std::vector;

//function to calculate the transformation matrix
vector<vector<float>> transform(vector<vector<float>> &Old, vector<vector<float>> &New){
    vector<vector<float>> inv(3,vector<float>(3));
    vector<vector<float>> trans(3,vector<float>(3));
    float det = Old[0][0] * (Old[1][1] * Old[2][2] - Old[2][1] * Old[1][2]) -
             Old[0][1] * (Old[1][0] * Old[2][2] - Old[1][2] * Old[2][0]) +
             Old[0][2] * (Old[1][0] * Old[2][1] - Old[1][1] * Old[2][0]); // calculating the determinant

    float rdet = 1 / det;
    
    inv[0][0] = (Old[1][1] * Old[2][2] - Old[2][1] * Old[1][2]) * rdet;
    inv[0][1] = (Old[0][2] * Old[2][1] - Old[0][1] * Old[2][2]) * rdet;
    inv[0][2] = (Old[0][1] * Old[1][2] - Old[0][2] * Old[1][1]) * rdet;
    inv[1][0] = (Old[1][2] * Old[2][0] - Old[1][0] * Old[2][2]) * rdet;
    inv[1][1] = (Old[0][0] * Old[2][2] - Old[0][2] * Old[2][0]) * rdet;
    inv[1][2] = (Old[1][0] * Old[0][2] - Old[0][0] * Old[1][2]) * rdet;
    inv[2][0] = (Old[1][0] * Old[2][1] - Old[2][0] * Old[1][1]) * rdet;
    inv[2][1] = (Old[2][0] * Old[0][1] - Old[0][0] * Old[2][1]) * rdet;
    inv[2][2] = (Old[0][0] * Old[1][1] - Old[1][0] * Old[0][1]) * rdet; // inverse of old matrix

    for(int i=0; i<3;i++){
        for(int j=0; j<3;j++){
            for(int k=0;k<3;k++){
                trans[i][j] += New[i][k]*inv[k][j]; // T = New*Old_inverse
            }
        }
    }
    return trans;
}

int main(){
    cv::Mat image;
    string filename;
    cout<<"Enter the name of image with extension"<<endl;
    cin>>filename;
    image = cv::imread(filename);
    int row = image.rows;
    int col = image.cols;
    vector<vector<float>> Old{{0,0,(float)row-1},{0,(float)col-1,0},{1,1,1}}; // pixel location of old matrix
    vector<vector<float>> New{{(float)0.33*row,(float)0.25*row,(float)0.7*row},{0,(float)0.85*col,(float)0.15*col},{1,1,1}}; // pixel location of new matrix
    vector<vector<float>> affineTrans = transform(Old,New); //calculating transformation matrix
    cv:: Mat dst = cv::Mat::zeros(image.size(),CV_8UC3);
    for(int i=0; i<row;i++){
        for(int j=0;j<col;j++){
                int x, y;
                //applying transformation on each pixel to calculate new pixel position
                x = (int)(affineTrans[0][0]*i+affineTrans[0][1]*j+affineTrans[0][2]*1); 
                y = (int)(affineTrans[1][0]*i+affineTrans[1][1]*j+affineTrans[1][2]*1);

                //copying pixel values from old position to new position
                dst.at<cv::Vec3b>(x,y)[0] = image.at<cv::Vec3b>(i,j)[0];
                dst.at<cv::Vec3b>(x,y)[1] = image.at<cv::Vec3b>(i,j)[1];
                dst.at<cv::Vec3b>(x,y)[2] = image.at<cv::Vec3b>(i,j)[2];
        }
    }
    cv::namedWindow("Warped image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Warped image",dst);
    cv::imwrite("warped.jpg",dst);
    cv::waitKey(0);
    return 0;
}