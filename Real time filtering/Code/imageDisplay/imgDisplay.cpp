#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using std::string;

int main(){
    cv::Mat image;
    string filename;
    cout<<"Enter the name of image with extension"<<endl;
    cin>>filename;
    cout<<"Press 'b' to blur the image"<<endl;
    cout<<"Press 'e' to find the edges of image"<<endl;
    std::cout<<"Press q to exit"<<std::endl;
    cv::namedWindow("Image",1);
    image = cv::imread(filename);

    while(true){
        cv::resize(image,image,cv::Size(550,700));
        cv::imshow("Image", image);
        char key = cv::waitKey(0);

        if (key == 'b')
            cv::GaussianBlur(image,image,cv::Size(5,5),1,1); 
        else if (key == 'e'){
            cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(image,image,cv::Size(5,5),1,1);
            cv::Canny(image,image,50,200); 
        }
        else if (key == 'q')
                break;
    }
    cv::destroyAllWindows();
    std::cout<<"Terminating Windows!"<<std::endl;
    return 0;
}