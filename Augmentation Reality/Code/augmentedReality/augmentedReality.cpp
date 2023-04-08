//Created by Rahul Kumar

//Code to implement the augmented reality
//It will project a virtual object onto a chessboard pattern

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include<vector>

using std::vector;
using std::string;

int main(){
    cv::VideoCapture vid;
    cv::Mat image;
    
    // Opening the video channel by streaming phone camera
    //vid.open(0);
    vid.open("http://10.0.0.26:4747/video");

    // Checking for successful opening of video port
    if (!vid.isOpened()){
        std::cout << "Alert! Camera is not accessible" << std::endl;
        return -1;
    }

    //Creating the windows to display video
    cv::namedWindow("Augmented Reality", cv::WINDOW_NORMAL);
    int count=1;

    //loading the intrinsic camera parameters
    cv::Mat camera_matrix, dist_coeff;
    
    //different camera matrix can be loaded
    //string filename = "webcam_Intrinsic_parameters.xml";
    string filename = "samsung_Intrinsic_parameters.xml";
    cv::FileStorage fs(filename,cv::FileStorage::READ);
    fs["camera_matrix"]>>camera_matrix;
    fs["distortion_coefficient"]>>dist_coeff;
    fs.release();

    float z = 0;
    
    while(true){
        vid.read(image);
        char key = cv::waitKey(2);
        char pattern1;

        if (key == 'a'){
            pattern1 = key;
        }
        else if (key == 'q'){
            pattern1 = key;
        }
        else if (key == 'b'){
            pattern1 = key;
        }

        //checking for successful reading of image frame
        if(image.empty()){
            std::cout<<"warning! No frame available to display"<<std::endl;
            break;
        }
        
        //defining the vector to store pixel coordinates and world coordinates
        vector<cv::Point2f> corner_set;
        vector<cv::Vec3f> point_set;
        bool patternFound;
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        //generating the corners of chessboard
        cv::Size patternSize(9, 6);
        patternFound = cv::findChessboardCorners(image, patternSize, corner_set, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
       
        if (patternFound){
            cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.0001));
        }
        
        if(!patternFound){
            continue;
        }
        
        //providing the world coordinates corresponding to the pixel coordinates of chessboard corners
        for(int i=0;i>-6;i--){
            for(int j=0;j<9;j++){
                point_set.push_back(cv::Vec3f(j,i,0));
            }
        }
        
        //Calculating the extrinsic parameters of camera
        cv::Mat rotVec,rotMat, trans;
        cv::solvePnP(point_set,corner_set,camera_matrix,dist_coeff,rotVec,trans);
        cv::Rodrigues(rotVec, rotMat);

        //To display the rotation and translation matrix
        // std::cout<<"Rotation matrix: "<<std::endl;
        // std::cout<<rotMat<<std::endl;
        // std::cout<<"Translational matrix: "<<std::endl;
        // std::cout<<trans<<std::endl;

        //Providing world coordinates for creating axis in the image plane
        vector<cv::Point3f> worldFrame;
        worldFrame.push_back(cv::Point3f(0,0,0));
        worldFrame.push_back(cv::Point3f(2,0,0));
        worldFrame.push_back(cv::Point3f(0,2,0));
        worldFrame.push_back(cv::Point3f(0,0,2));

        //Projecting 3D coordinates into 2D coordinates
        vector<cv::Point2f> cameraFrame;
        cv::projectPoints(worldFrame,rotMat, trans, camera_matrix,dist_coeff, cameraFrame);

        //drawing line using the projected coordinates of world point
        cv::line(image,cameraFrame[0],cameraFrame[1], cv::Scalar(255,0,0),2);
        cv::line(image,cameraFrame[0],cameraFrame[2], cv::Scalar(0,255,0),2);
        cv::line(image,cameraFrame[0],cameraFrame[3], cv::Scalar(0,0,255),2);

        //For projecting a robot wireframe on the image plane and providing world coordinates for robot
        if(pattern1 == 'a'){
            vector<cv::Point3f> worldCord;
            worldCord.push_back(cv::Point3f(1,-1,0+z));
            worldCord.push_back(cv::Point3f(3,-1,0+z));
            worldCord.push_back(cv::Point3f(3,-2,0+z));
            worldCord.push_back(cv::Point3f(1,-2,0+z));
            worldCord.push_back(cv::Point3f(1,-3,0+z));
            worldCord.push_back(cv::Point3f(3,-3,0+z));
            worldCord.push_back(cv::Point3f(3,-4,0+z));
            worldCord.push_back(cv::Point3f(1,-4,0+z));
            worldCord.push_back(cv::Point3f(3,-1,0.5+z));
            worldCord.push_back(cv::Point3f(3,-2,0.5+z));
            worldCord.push_back(cv::Point3f(2,-2,0.5+z));
            worldCord.push_back(cv::Point3f(2,-1,0.5+z));
            worldCord.push_back(cv::Point3f(3,-3,0.5+z));
            worldCord.push_back(cv::Point3f(3,-4,0.5+z));
            worldCord.push_back(cv::Point3f(2,-4,0.5+z));
            worldCord.push_back(cv::Point3f(2,-3,0.5+z));
            worldCord.push_back(cv::Point3f(1,-1,4+z));
            worldCord.push_back(cv::Point3f(2,-1,4+z));
            worldCord.push_back(cv::Point3f(2,-2,4+z));
            worldCord.push_back(cv::Point3f(1,-2,4+z));
            worldCord.push_back(cv::Point3f(1,-3,4+z));
            worldCord.push_back(cv::Point3f(2,-3,4+z));
            worldCord.push_back(cv::Point3f(2,-4,4+z));
            worldCord.push_back(cv::Point3f(1,-4,4+z));
            //Chest
            worldCord.push_back(cv::Point3f(1,-1,7+z));
            worldCord.push_back(cv::Point3f(2,-1,7+z));
            worldCord.push_back(cv::Point3f(2,-4,7+z));
            worldCord.push_back(cv::Point3f(1,-4,7+z));
            //Head
            worldCord.push_back(cv::Point3f(1,-1.75,7+z));
            worldCord.push_back(cv::Point3f(3,-1.75,7+z));
            worldCord.push_back(cv::Point3f(3,-3.25,7+z));
            worldCord.push_back(cv::Point3f(1,-3.25,7+z));
            worldCord.push_back(cv::Point3f(1,-1.75,9+z));
            worldCord.push_back(cv::Point3f(3,-1.75,9+z));
            worldCord.push_back(cv::Point3f(3,-3.25,9+z));
            worldCord.push_back(cv::Point3f(1,-3.25,9+z));
            //Right Hand
            worldCord.push_back(cv::Point3f(2,-5,7+z));
            worldCord.push_back(cv::Point3f(2,-5,6+z));
            worldCord.push_back(cv::Point3f(2,-4,6+z));
            worldCord.push_back(cv::Point3f(1,-5,7+z));
            worldCord.push_back(cv::Point3f(1,-5,6+z));
            worldCord.push_back(cv::Point3f(1,-4,6+z));
            worldCord.push_back(cv::Point3f(5,-4,7+z));
            worldCord.push_back(cv::Point3f(5,-5,7+z));
            worldCord.push_back(cv::Point3f(5,-5,6+z));
            worldCord.push_back(cv::Point3f(5,-4,6+z));
            //Left Hand
            worldCord.push_back(cv::Point3f(1,0,7+z));
            worldCord.push_back(cv::Point3f(1,0,6+z));
            worldCord.push_back(cv::Point3f(1,-1,6+z));
            worldCord.push_back(cv::Point3f(2,0,7+z));
            worldCord.push_back(cv::Point3f(2,0,6+z));
            worldCord.push_back(cv::Point3f(2,-1,6+z));
            worldCord.push_back(cv::Point3f(5,-1,7+z));
            worldCord.push_back(cv::Point3f(5,0,7+z));
            worldCord.push_back(cv::Point3f(5,0,6+z));
            worldCord.push_back(cv::Point3f(5,-1,6+z));

            z = z+0.025; //can be updated to provide the motion of robot in z direction

            //projecting the points on 2D plane and drawing the robot
            vector<cv::Point2f> cameraCord;
            cv::projectPoints(worldCord,rotMat, trans, camera_matrix,dist_coeff, cameraCord);
            //Foot
            cv::line(image,cameraCord[0],cameraCord[1], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[1],cameraCord[2], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[2],cameraCord[3], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[3],cameraCord[0], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[4],cameraCord[5], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[5],cameraCord[6], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[6],cameraCord[7], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[7],cameraCord[4], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[1],cameraCord[8], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[2],cameraCord[9], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[8],cameraCord[9], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[9],cameraCord[10], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[10],cameraCord[11], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[11],cameraCord[8], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[5],cameraCord[12], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[6],cameraCord[13], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[12],cameraCord[13], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[13],cameraCord[14], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[14],cameraCord[15], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[15],cameraCord[12], cv::Scalar(255,0,0),2);

            //Leg
            cv::line(image,cameraCord[0],cameraCord[16], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[3],cameraCord[19], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[11],cameraCord[17], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[10],cameraCord[18], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[16],cameraCord[17], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[17],cameraCord[18], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[18],cameraCord[19], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[19],cameraCord[16], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[4],cameraCord[20], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[15],cameraCord[21], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[14],cameraCord[22], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[7],cameraCord[23], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[20],cameraCord[21], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[21],cameraCord[22], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[22],cameraCord[23], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[23],cameraCord[20], cv::Scalar(255,0,0),2);
            //chest
            cv::line(image,cameraCord[16],cameraCord[24], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[17],cameraCord[25], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[22],cameraCord[26], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[23],cameraCord[27], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[24],cameraCord[25], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[25],cameraCord[26], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[26],cameraCord[27], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[27],cameraCord[24], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[16],cameraCord[23], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[17],cameraCord[22], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[27],cameraCord[24], cv::Scalar(255,255,0),2);
            cv::line(image,cameraCord[27],cameraCord[24], cv::Scalar(255,255,0),2);
            //Head
            cv::line(image,cameraCord[28],cameraCord[29], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[29],cameraCord[30], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[30],cameraCord[31], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[31],cameraCord[28], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[28],cameraCord[32], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[29],cameraCord[33], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[30],cameraCord[34], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[31],cameraCord[35], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[32],cameraCord[33], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[33],cameraCord[34], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[34],cameraCord[35], cv::Scalar(255,0,255),2);
            cv::line(image,cameraCord[35],cameraCord[32], cv::Scalar(255,0,255),2);
            //right hand
            cv::line(image,cameraCord[26],cameraCord[36], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[36],cameraCord[37], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[37],cameraCord[38], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[38],cameraCord[26], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[27],cameraCord[39], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[39],cameraCord[40], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[40],cameraCord[41], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[41],cameraCord[27], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[26],cameraCord[27], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[36],cameraCord[39], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[37],cameraCord[40], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[38],cameraCord[41], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[26],cameraCord[42], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[36],cameraCord[43], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[37],cameraCord[44], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[38],cameraCord[45], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[42],cameraCord[43], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[43],cameraCord[44], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[44],cameraCord[45], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[45],cameraCord[42], cv::Scalar(255,0,0),2);
            //left hand
            cv::line(image,cameraCord[24],cameraCord[46], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[46],cameraCord[47], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[47],cameraCord[48], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[48],cameraCord[24], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[25],cameraCord[49], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[49],cameraCord[50], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[50],cameraCord[51], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[51],cameraCord[25], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[24],cameraCord[25], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[46],cameraCord[49], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[47],cameraCord[50], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[48],cameraCord[51], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[25],cameraCord[52], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[49],cameraCord[53], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[50],cameraCord[54], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[51],cameraCord[55], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[52],cameraCord[53], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[53],cameraCord[54], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[54],cameraCord[55], cv::Scalar(255,0,0),2);
            cv::line(image,cameraCord[55],cameraCord[52], cv::Scalar(255,0,0),2);
        }

        //displaying the video of augmented reality
        cv::resizeWindow("Augmented Reality", cv::Size(720, 540));
        cv::imshow("Augmented Reality", image);

        // Terminating the video if 'q' is pressed
        if (key == 'q'){
            std::cout<<"Terminating camera feed!"<<std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}