#define CAFFE
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>

TrackingManager tm;

uint64_t people = 0;

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  Source s("f:/my.mp4");

  if(!s.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  cv::namedWindow("Motion Capture");

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

    auto scale = (float) 600 / frame.cols;

    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);

    cv::Mat gray, blur, delta, thresh, dilate;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, blur, cv::Size(21,21), 0);

    if (ff.empty())
    {
      ff = blur;
    }

	  cv::absdiff(ff, blur, delta);

	  cv::threshold(delta, thresh, 25, 255, cv::THRESH_BINARY);

    cv::dilate(thresh, dilate, cv::Mat(), cv::Point(-1, -1), 2);

    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten

    // cv::Mat contourOutput = dilate.clone();

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(dilate, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) 
    {
      auto bb = cv::boundingRect(contours[i]);
      
      if (cv::contourArea(contours[i]) < 10000) continue;
      if (bb.width > bb.height) continue;

      /*
       * skip this contour if:
       * it is inside any other bb
       * A is less than the threshold
       */
      for (size_t j = 0; (j < contours.size() && (i != j)); j++)
      {
        if (IsRectInsideRect(bb, cv::boundingRect(contours[j])))
        {
          break;
        }
        else
        {
          auto rv = cv::matchShapes(contours[i], contours[j], 1, 0.0);
          
          if (rv < 0.3) std::cout << rv << "\n";

          if (1)
          {
            people++;
            cv::rectangle(frame, bb, cv::Scalar(255, 0, 0 ), 1, 1);            
          }
        }
      }
    }

    /*
     * update all active trackers first
     */

    cv::putText(frame, std::to_string(people), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("Motion Capture", frame);
    cv::imshow("Mask", delta);

    if (!s.HandleUserInput()) break;
  }   
}
