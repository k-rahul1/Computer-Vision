//Created by Rahul Kumar

//Code to implement the real time recognition of 2D objects

#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include "filter.h"
#include"csv_util.h"
#include <vector>

using std::string;
using std::vector;

int main(){
    cv::VideoCapture vid;
    cv::Mat imgFrame,labelFrame; 

    // Opening the video channel by streaming phone camera
    vid.open("http://10.0.0.26:4747/video");

    // Checking for successful opening of video port
    if(!vid.isOpened()){
        std::cout<<"Alert! Camera is not accessible"<<std::endl;
        return -1;
    }
    int i =1;
    char featureFile[256] = "LabeledFeature.csv";

    //Creating the windows to display video
    cv::namedWindow("Live Video",cv::WINDOW_NORMAL);
    cv::namedWindow("Threshold Video",cv::WINDOW_NORMAL);
    cv::namedWindow("CleanUp Video",cv::WINDOW_NORMAL);
    cv::namedWindow("Segment Video",cv::WINDOW_NORMAL);
    cv::namedWindow("Feature Video",cv::WINDOW_NORMAL);
    cv::namedWindow("Labeled object Video",cv::WINDOW_NORMAL);

    char NNclassification, KNNclassification;

    std::cout<<"Press 'n' to add new label to the database"<<std::endl;
    std::cout<<"Press 'c' to classify using Nearest neighbor"<<std::endl;
    std::cout<<"Press 'k' to classify using K-Nearest neighbor"<<std::endl;

    // Creating the loop to display the video frame by frame
    while(true){
        vid.read(imgFrame);
        labelFrame = imgFrame.clone();
        char key = cv::waitKey(1);
        if(key == 'c'){
            NNclassification = KNNclassification = key;
        }
        else if(key == 'k'){
            NNclassification = KNNclassification = key;
        }
        else if(key == 'n'){
            NNclassification = KNNclassification = key;
        }
        
        //Applying the thresholding to video to separate foreground from background 
        cv::Mat thresholdFrame;
        thresholding(imgFrame,thresholdFrame);

        //Cleaning up the thresholded video to fill the holes and remove the noise
        cv::Mat cleanFrame;
        clean(thresholdFrame,cleanFrame);

        //Segmenting the foreground to detect different contours in foreground
        cv::Mat segmentFrame;
        vector<int> reqRegionId = Segment(cleanFrame,segmentFrame);

        //Generating the feature vector of contours
        cv::Mat label,stat,centroid, featureFrame;
        int regionId = cv::connectedComponentsWithStats(cleanFrame,label,stat,centroid,8);
        vector<float> featureVec = Feature(segmentFrame,featureFrame,label,regionId,reqRegionId);
        
        //Key activated code to create csv file of labeled objects with feature vector
        if(key=='n'){
            std::cout<<"Enter the label of object"<<std::endl;
            char name[256];
            std::cin>>name;
            append_image_data_csv(featureFile,name, featureVec, 0 );
        }

        //Key activated to classify the objects using Nearest Neighbor classification
        if(NNclassification=='c'){
            char featureFile[256] = "LabeledFeature.csv";
            std::vector<char *> labelnames;
            std::vector<std::vector<float>> data;

            //reading image name and feature vector from already saved csv file
            read_image_data_csv( featureFile, labelnames, data, 0 );

            //Generating the feature vector of unknown object
            vector<float> objfeature = Feature(segmentFrame,featureFrame,label,regionId,reqRegionId);

            //calculating the distance of unknown object from the labeled objects using Scaled Euclidean
            vector<float> distance = distanceMetric(objfeature, data);

            //generating pair of distance and corresponding image
            std::vector<std::pair<float,char *>> DistImgPair;
            for(int i=0;i<(int)data.size();i++){
                DistImgPair.push_back(std::make_pair(distance[i],labelnames[i])); 
            }
            sort(DistImgPair.begin(),DistImgPair.end()); //sorting as per distance
            cv::Point P;
            P.x = stat.at<int>(cv::Point(0,reqRegionId[0]));
            P.y = stat.at<int>(cv::Point(1,reqRegionId[0]));

            if(DistImgPair[0].first>1){
                cv::putText(labelFrame,"Unknown",P,cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(255,0,0),1);
                std::cout<<"Press 'n' and enter the label of unknown object"<<std::endl;
            }
            else{
                //Displaying the label of unknown object
                cv::putText(labelFrame,DistImgPair[0].second,P,cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(255,0,0),1);
            }

        }


        //Key activated to classify the objects using K-Nearest Neighbor classification
        if(KNNclassification=='k'){
            char featureFile[256] = "LabeledFeature.csv";
            std::vector<char *> labelnames;
            std::vector<std::vector<float>> data;

            //reading image name and feature vector from already saved csv file
            read_image_data_csv( featureFile, labelnames, data, 0 );

            //Generating the feature vector of unknown object
            vector<float> objfeature = Feature(segmentFrame,featureFrame,label,regionId,reqRegionId);

            //calculating the distance of unknown object from the labeled objects using Scaled Euclidean
            vector<float> distance = distanceMetric(objfeature, data);

            char uniqfeatureFile[256] = "uniqueFeature.csv";
            std::vector<char *> uniqlabelnames;
            std::vector<std::vector<float>> nulldata;
            read_image_data_csv( uniqfeatureFile, uniqlabelnames, nulldata, 0 ); //reading the label of unique classes

            vector<float> KNNdistance(uniqlabelnames.size());
            vector<float> classdistance;
            for(int i=0; i<uniqlabelnames.size();i++){
                for(int j=0; j<distance.size(); j++){
                    if(!strcmp(labelnames[j],uniqlabelnames[i])){
                        classdistance.push_back(distance[j]);  //generating vector of distance from labeled objects of each class
                    }
                }

                sort(classdistance.begin(),classdistance.end()); //sorting the distance of labeled objects of each class
                KNNdistance[i] = classdistance[0]+classdistance[1]+classdistance[2]; //3-nearest neighbor distance from each class
                classdistance.clear();
            }

            //generating pair of distance and corresponding image
            std::vector<std::pair<float,string>> DistImgPair;
            for(int i=0;i<(int)KNNdistance.size();i++){
                DistImgPair.push_back(std::make_pair(KNNdistance[i],uniqlabelnames[i])); 
            }

            sort(DistImgPair.begin(),DistImgPair.end()); //sorting as per distance

            cv::Point P;
            P.x = stat.at<int>(cv::Point(0,reqRegionId[0]));
            P.y = stat.at<int>(cv::Point(1,reqRegionId[0]));

            //Displaying the label of unknown object
            cv::putText(labelFrame,DistImgPair[0].second,P,cv::FONT_HERSHEY_COMPLEX_SMALL,1,cv::Scalar(255,0,0),1);
        }


        //checking for successful reading of image frame
        if(imgFrame.empty()){
            std::cout<<"warning! No frame available to display"<<std::endl;
            break;
        }
        
        cv::resizeWindow("Live Video",cv::Size(720,720));
        cv::resizeWindow("Threshold Video",cv::Size(720,720));
        cv::resizeWindow("CleanUp Video",cv::Size(720,720));
        cv::resizeWindow("Segment Video",cv::Size(720,720));
        cv::resizeWindow("Feature Video",cv::Size(720,720));
        cv::resizeWindow("Labeled object Video",cv::Size(720,720));

        //Displaying the video of each step of object recognition
        cv::imshow("Live Video", imgFrame);
        cv::imshow("Threshold Video", thresholdFrame);
        cv::imshow("CleanUp Video", cleanFrame);
        cv::imshow("Segment Video", segmentFrame);
        cv::imshow("Feature Video", featureFrame);
        cv::imshow("Labeled object Video", labelFrame);
        //char key = cv::waitKey(5);

        string filename = "image" + std::to_string(i) + ".jpg";
        // Saving the images once 's' is pressed
        if(key =='s'){
            cv::imwrite(filename,labelFrame);
            i++;
        }

        // Terminating the webcam if 'q' is pressed
        if(key =='q'){
            std::cout<<"Terminating camera feed!"<<std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}