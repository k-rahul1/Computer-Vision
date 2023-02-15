#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>

using std::string;
using namespace std;

int main(){
    cv::Mat image;
    string filename;
    cout<<"Enter the name of image with extension"<<endl;
    cin>>filename;
    image = cv::imread(filename);
    char key;
    while(key!='q'){
        //taking text and pixel location from user
        int x,y;
        cout<<"Enter x position to put text on image between 0 and "<<image.cols<<endl;
        cin>>x;
        cout<<"Enter y position to put text on image between 0 and "<<image.rows<<endl;
        cin>>y;
        cin.ignore();
        string text;
        cout<<"Enter the text"<<endl;
        getline(cin,text);

        //checking for valid input
        if(x>=image.cols || y>=image.rows){
            cout<<"Wrong pixel position is entered!"<<endl;
            break;
        }

        //Putting text on image
        cv::putText(image,text,cv::Point(x,y),cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,0),2);
        cv::putText(image,text,cv::Point(x,y),cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(0,0,255),1);

        //option for multiple entry of text on image
        cout<<"Press 'c' to continue insert text or press 'q' to see the result"<<endl;
        cin>>key;
    }
    cv::imshow("Meme",image);
    cout<<"Press 's' to save image or press 'e' to exit without saving"<<endl;
    char input = cv::waitKey(0);
    if(input=='s'){
        cv::imwrite("Meme.jpg",image);
    }
    return 0;
}