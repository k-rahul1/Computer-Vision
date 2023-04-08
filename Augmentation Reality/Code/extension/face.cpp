//Created by Rahul Kumar

//Code to detect the face and augment sunglasses on face in real time

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>

using std::string;

int count =1;

int main(){
    // Initialize video capture
    cv::VideoCapture vid;

    //vid.open(0);
    vid.open("http://10.0.0.26:4747/video");

    if (!vid.isOpened()){
        std::cout << "Cannot open video capture" << std::endl;
        return -1;
    }

    // Reading the image of sunglasses
    cv::Mat imgGlassesOriginal = cv::imread("Sunglasses1.png",-1);

    // Creating a mask for sunglasses with alpha channel
    cv::Mat maskGlassesOriginal;
    cv::extractChannel(imgGlassesOriginal, maskGlassesOriginal, 3);

    // removing the alpha channel from the image of sunglasses
    imgGlassesOriginal = imgGlassesOriginal(cv::Rect(0, 0, imgGlassesOriginal.cols, imgGlassesOriginal.rows));
    cv::cvtColor(imgGlassesOriginal, imgGlassesOriginal, cv::COLOR_BGRA2BGR);

    // defining the objects of CascadeClassifier class for detecting face and eye
    cv::CascadeClassifier face_detector;
    face_detector.load("haarcascade_frontalface_default.xml");
    cv::CascadeClassifier eye_detector;
    eye_detector.load("haarcascade_mcs_eyepair_big.xml");

    cv::namedWindow("Augmented face", cv::WINDOW_NORMAL);

    // Loop through video frames
    cv::Mat frame;
    while (true){
        vid.read(frame);

        // Detect faces
        std::vector<cv::Rect> face_boxes;

        face_detector.detectMultiScale(frame, face_boxes);
        if (face_boxes.size() < 1){
            cv::resizeWindow("Augmented face", cv::Size(720, 720));
            cv::imshow("Augmented face", frame);
        }
        else{
            // looping through all the faces detected
            for (const cv::Rect &face_box : face_boxes){
                int fx1 = face_box.x;
                int fy1 = face_box.y;
                int fw = face_box.width;
                int fh = face_box.height;
                int fx2 = fx1 + fw;
                int fy2 = fy1 + fh;
                cv::Mat face = frame.rowRange(fy1, fy2).colRange(fx1, fx2);

                // eye section
                std::vector<cv::Rect> eye_boxes;
                eye_detector.detectMultiScale(face, eye_boxes);
                if (eye_boxes.empty())
                {
                    continue;
                }
                cv::Rect eye_box = eye_boxes[0];
                int ex1 = eye_box.x;
                int ey1 = eye_box.y;
                int ew = eye_box.width;
                int eh = eye_box.height;
                int ex2 = ex1 + ew;
                int ey2 = ey1 + eh;

                // Sunglasses section
                int gx1 = ex1 - ew / 8;
                int gx2 = ex2 + ew / 8;
                int gy1 = ey1;
                int gy2 = ey2 + 0.5 * eh;

                // defining the feasible boundary for sunglasses
                gy1 = (gy1 > 0) ? gy1 : 0;
                gy2 = (gy2 < fh) ? gy2 : fh;
                gx1 = (gx1 > 0) ? gx1 : 0;
                gx2 = (gx2 < fw) ? gx2 : fw;

                int gw = std::abs(gx2 - gx1);
                int gh = std::abs(gy2 - gy1);

                // resizing the sunglasses image as well as mask as per the feasible boundary on face
                cv::Mat imageGlasses;
                resize(imgGlassesOriginal, imageGlasses, cv::Size(gw, gh));
                cv::Mat maskGlasses;
                resize(maskGlassesOriginal, maskGlasses, cv::Size(gw, gh));

                // creating a mat file to store the roi of face for sunglasses
                cv::Mat face_glasses = face(cv::Rect(gx1, gy1, gw, gh));

                // creating an inverse binary mask from mask of sunglasses
                cv::Mat inv_glasses_mask;
                bitwise_not(maskGlasses, inv_glasses_mask);
                inv_glasses_mask.convertTo(inv_glasses_mask, CV_8UC1);

                // retaining the region of face outside of glasses with inverse mask
                cv::Mat background_face;
                bitwise_and(face_glasses, face_glasses, background_face, inv_glasses_mask);

                // retaining the region of only sunglasses and discarding pixels that lie on face with mask Glasses
                cv::Mat foreground_glasses;
                bitwise_and(imageGlasses, imageGlasses, foreground_glasses, maskGlasses);

                // combining the background and foreground region into a single mat file
                cv::Mat glasses;
                add(background_face, foreground_glasses, glasses);

                // creating a region of interest and copying the glasses on face
                cv::Mat face_roi = face(cv::Rect(gx1, gy1, gw, gh));
                glasses.copyTo(face_roi);

                // creating a rectangular frame to show the tracked face
                rectangle(frame, cv::Point2f(fx1, fy1), cv::Point2f(fx2, fy2), cv::Scalar(0, 255, 0), 2);
            //}
            // displaying the augmented face
            cv::resizeWindow("Augmented face", cv::Size(720, 720));
            cv::imshow("Augmented face", frame);
        }
        }

        // Break the loop if q is pressed
        if (cv::waitKey(2) == 'q'){
            std::cout<<"Terminating camera feed!"<<std::endl;
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
