#ifndef TRACKER_HPP
#define TRACKER_HPP 

  //Haar
  // if(!face_cascade.load("C:\\code\\my\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
  // {
  //   std::cout << "Error loading face cascade\n";
  //   return -1;
  // };
  // if(!eyes_cascade.load("C:\\code\\my\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"))
  // {
  //   std::cout << "Error loading eye cascade\n";
  //   return -1;
  // };

// void DetectFacesHaar(cv::Mat& frame)
// {
//     cv::Mat frame_gray;
//     cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
//     equalizeHist(frame_gray, frame_gray);

//     std::vector<cv::Rect> faces;

//     face_cascade.detectMultiScale(frame_gray, faces);

//     for (size_t i = 0; i < faces.size(); i++)
//     {
//       // cv::Mat faceROI = frame_gray(faces[i]);
//       // std::vector<cv::Rect> eyes;
//       // eyes_cascade.detectMultiScale(faceROI, eyes);
//       // if (eyes.size())
//       // {
//       // }
//       cv::rectangle(
//         frame,
//         cv::Point(faces[i].x, faces[i].y),
//         cv::Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//         cv::Scalar(0, 255, 0), 2, 4);      
//     }
// }

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

class DDetector
{
    enum Mode { Default, Daimler } m;
    HOGDescriptor hog, hog_d;
public:
    DDetector() : m(Default), hog(), hog_d(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    vector<Rect> detect(InputArray img)
    {
        // Run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        vector<Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, Size(8,8), Size(), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0, Size(8,8), Size(), 1.05, 2, true);
        return found;
    }
    void adjustRect(Rect & r) const
    {
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
};

static const string keys = "{ help h   |   | print help message }"
                           "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                           "{ video v  |   | use video as input }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the use of the HoG descriptor.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string file = "C:/code/my.mp4"; //parser.get<string>("video");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if (file.empty())
        cap.open(camera);
    else
    {
        file = samples::findFileOrKeep(file);
        cap.open(file);
    }
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }

    cout << "Press 'q' or <ESC> to quit." << endl;
    cout << "Press <space> to toggle between Default and Daimler detector" << endl;
    DDetector detector;
    Mat frame;
    for (;;)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
    auto scale = (float) 600 / frame.cols;
    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);        
        int64 t = getTickCount();
        vector<Rect> found = detector.detect(frame);
        t = getTickCount() - t;

        // show the window
        {
            ostringstream buf;
            buf << "Mode: " << detector.modeName() << " ||| "
                << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }
        for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
        {
            Rect &r = *i;
            detector.adjustRect(r);
            rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
        }
        imshow("People detector", frame);

        // interact with user
        const char key = (char)waitKey(1);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == ' ')
        {
            detector.toggleMode();
        }
    }
    return 0;
}


#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) 
{   
    cv::namedWindow("Annotated Frame");
    cv::namedWindow("Contour Delta");
   
    cv::VideoCapture capture("c:\\code\\walk.mp4"); //rtsp://admin:Appo7ite@192.168.0.201:554/Streaming/Channels/1");

    int frameIndex = 0;
    Mat lastFrame;
    cv::VideoWriter writer;
    
    while ( capture.isOpened() )     // check !!
    {
        cv::Mat frame;
        if ( ! capture.read(frame) ) // another check !!
            break;
        
#if 1
        Mat grayFrame, dilatedFrame, edges, deltaFrame, deltaCopyFrame;
        
        // scale down image
        cv::resize(frame, frame, Size(0,0), 0.33, 0.33);

        // convert to grayscale
        cvtColor(frame, grayFrame, CV_BGR2GRAY);
        
        // add blur
        GaussianBlur(grayFrame, grayFrame, Size(21,21), 0);

        if (frameIndex == 0) {
            frameIndex++;
            
            // position the windows
            moveWindow("Annotated Frame", 0, 0);
            moveWindow("Contour Delta", 0, grayFrame.size().height);
            
            std::cout << "FRAME SIZE = " << grayFrame.size().width << " x " << grayFrame.size().height << "\n";

            lastFrame = grayFrame;
            continue;
        } else if ((frameIndex % 50) == 0) {
            frameIndex = 0;
        }
        frameIndex++;
        
        
        // create difference frame
        cv::absdiff(lastFrame, grayFrame, deltaFrame);
        cv::threshold(deltaFrame, deltaFrame, 50, 255, cv::THRESH_BINARY);
        
        // dilate to fill-in holes and find contours
        int iterations = 2;
        cv::dilate(deltaFrame, deltaFrame, Mat(), Point(-1,-1), iterations);
      
        /// Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        //deltaFrame.copyTo(deltaCopyFrame);
        cv::findContours(deltaFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );
        
        for( int i = 0; i < contours.size(); i++ )
        { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }
        
        
        /// Draw polygonal contour + bonding rects + circles
        //Mat drawing = Mat::zeros( deltaFrame.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar(255, 0, 0);
            drawContours( frame, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
            rectangle( frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
            circle( frame, center[i], (int)radius[i], color, 2, 8, 0 );
        }

        // find edges with canny
        //Canny(blurFrame, edges, 0, 30, 3);
        
        imshow("Annotated Frame", frame);
        //imshow("Edges", edges);
        imshow("Contour Delta", deltaFrame);
#else
        imshow("Frame", frame);
#endif
        
        // write the frame to the video file
        //std::cout << "WRITING FRAME...";
        writer.write(frame);

        // wait for escape (needed to exit and properly write the video file)
        switch(waitKey(1)) {
            case 27:
                capture.release();
                writer.release();
                return 0;
        }
    }
    return 0;
    
}

#endif //TRACKER_HPP