#define CAFFE
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>

TrackingManager tm;

uint64_t people = 0;

int slider_ar = 100;

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  Source s(0);

  if(!s.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  cv::namedWindow("Motion Capture");

  cv::createTrackbar("w/h", "Motion Capture", &slider_ar, 200, nullptr);

  cv::Mat ff, frame;

  for (;;)
  {
    if(!s.Read(frame))
	  {
      if(s.HasEnded())
      {
        s.Rewind();
        tm.ClearAllContexts();
        people = 0;
        continue;
      }
      else
      {
        break;
      }
    }

    cv::flip(frame, frame, 1);

    auto scale = (float) 600 / frame.cols;

    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);

    cv::Mat gray, blur, delta, thresh, dilate;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, blur, cv::Size(21,21), 0);

    if (ff.empty())
    {
      ff = blur;
    }

	  //compute the absolute difference between
	  //the current frame and first frame

	  cv::absdiff(ff, blur, delta);

	  cv::threshold(delta, thresh, 25, 255, cv::THRESH_BINARY);

    cv::dilate(thresh, dilate, cv::Mat(), cv::Point(-1, -1), 2);

    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat contourOutput = dilate.clone();
    
    cv::findContours(contourOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat contourImage(frame.size(), CV_8UC3, cv::Scalar(0,0,0));

    for (size_t i = 0; i < contours.size(); i++) 
    {
      auto area = cv::contourArea(contours[i]);
      auto bb = cv::boundingRect(contours[i]);
      auto ar = (double)bb.width/bb.height;

      if (area > 10000 && (ar <= ((double)slider_ar/200)))
      {
        cv::rectangle(frame, bb, cv::Scalar(255, 0, 0 ), 1, 1);
        cv::putText(frame, "w/h : " + std::to_string(bb.width) + "/" + std::to_string(bb.height) +  ", A : " + std::to_string(area), bb.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
      }

      std::cout << "contour area : " << area << " aspect ration : " << std::to_string(ar) << "\n";
    }

    /*
     * update all active trackers first
     */

    cv::putText(frame, std::to_string(people), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("Motion Capture", frame);

    if (!s.HandleUserInput()) break;
  }   
}


// #include <iostream>
// #include "opencv2/opencv.hpp"

// using namespace cv;
// using namespace std;

// int main(int argc, const char * argv[]) 
// {   
//     cv::namedWindow("Annotated Frame");
//     cv::namedWindow("Contour Delta");
   
//     cv::VideoCapture capture("c:\\code\\walk.mp4"); //rtsp://admin:Appo7ite@192.168.0.201:554/Streaming/Channels/1");

//     int frameIndex = 0;
//     Mat lastFrame;
//     cv::VideoWriter writer;
    
//     while ( capture.isOpened() )     // check !!
//     {
//         cv::Mat frame;
//         if ( ! capture.read(frame) ) // another check !!
//             break;
        
// #if 1
//         Mat grayFrame, dilatedFrame, edges, deltaFrame, deltaCopyFrame;
        
//         // scale down image
//         cv::resize(frame, frame, Size(0,0), 0.33, 0.33);

//         // convert to grayscale
//         cvtColor(frame, grayFrame, CV_BGR2GRAY);
        
//         // add blur
//         GaussianBlur(grayFrame, grayFrame, Size(21,21), 0);

//         if (frameIndex == 0) {
//             frameIndex++;
            
//             // position the windows
//             moveWindow("Annotated Frame", 0, 0);
//             moveWindow("Contour Delta", 0, grayFrame.size().height);
            
//             std::cout << "FRAME SIZE = " << grayFrame.size().width << " x " << grayFrame.size().height << "\n";

//             lastFrame = grayFrame;
//             continue;
//         } else if ((frameIndex % 50) == 0) {
//             frameIndex = 0;
//         }
//         frameIndex++;
        
        
//         // create difference frame
//         cv::absdiff(lastFrame, grayFrame, deltaFrame);
//         cv::threshold(deltaFrame, deltaFrame, 50, 255, cv::THRESH_BINARY);
        
//         // dilate to fill-in holes and find contours
//         int iterations = 2;
//         cv::dilate(deltaFrame, deltaFrame, Mat(), Point(-1,-1), iterations);
      
//         /// Approximate contours to polygons + get bounding rects and circles
//         vector<vector<Point>> contours;
//         vector<Vec4i> hierarchy;
//         //deltaFrame.copyTo(deltaCopyFrame);
//         cv::findContours(deltaFrame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
//         vector<vector<Point> > contours_poly( contours.size() );
//         vector<Rect> boundRect( contours.size() );
//         vector<Point2f>center( contours.size() );
//         vector<float>radius( contours.size() );
        
//         for( int i = 0; i < contours.size(); i++ )
//         { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
//             boundRect[i] = boundingRect( Mat(contours_poly[i]) );
//             minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
//         }
        
        
//         /// Draw polygonal contour + bonding rects + circles
//         //Mat drawing = Mat::zeros( deltaFrame.size(), CV_8UC3 );
//         for( int i = 0; i< contours.size(); i++ )
//         {
//             Scalar color = Scalar(255, 0, 0);
//             drawContours( frame, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//             rectangle( frame, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//             circle( frame, center[i], (int)radius[i], color, 2, 8, 0 );
//         }

//         // find edges with canny
//         //Canny(blurFrame, edges, 0, 30, 3);
        
//         imshow("Annotated Frame", frame);
//         //imshow("Edges", edges);
//         imshow("Contour Delta", deltaFrame);
// #else
//         imshow("Frame", frame);
// #endif
        
//         // write the frame to the video file
//         //std::cout << "WRITING FRAME...";
//         writer.write(frame);

//         // wait for escape (needed to exit and properly write the video file)
//         switch(waitKey(1)) {
//             case 27:
//                 capture.release();
//                 writer.release();
//                 return 0;
//         }
//     }
//     return 0;
    
// }