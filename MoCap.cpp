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

struct TrackerContext
{
  std::vector<cv::Rect2d> trail; // last bb
  cv::Ptr<cv::Tracker> tracker;  // tracker 
};

std::vector<TrackerContext> Trackers;

uint64_t people = 0;
uint64_t count = 0;

int slider_ar = 100;

void on_trackbar(int x, void *y)
{

}

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  cv::VideoCapture cap("c:/code/calib.mp4"); //, cv::CAP_FFMPEG); 0, cv::CAP_DSHOW ); //P1033658 

  if(!cap.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  cv::namedWindow("Motion Capture");

  cv::createTrackbar("w/h", "Motion Capture", &slider_ar, 200, on_trackbar);

  cv::Mat ff, frame;

  uint64_t nTotal = cap.get(cv::CAP_PROP_FRAME_COUNT);

  for (;;)
  {
    if(!cap.read(frame))
	  {
      if(count == nTotal)
      {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        Trackers.clear();
        count = people = 0;
        continue;
      }
      else
      {
        break;
      }
    }

    count++;

    cv::Mat gray, blur, delta, thresh, dilate;

    auto scale = (float) 600 / frame.cols;

    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);

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
        cv::putText(frame, "w/h : " + std::to_string(bb.width) + "/" + std::to_string(bb.height) +  ", A : " + std::to_string(area), bb.br(), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
      }

      std::cout << "contour area : " << area << " aspect ration : " << std::to_string(ar) << "\n";
    }

    /*
     * update all active trackers first
     */
    for (int i = (Trackers.size() - 1); i >=0; i--)
    {
      auto& tc = Trackers[i];

      cv::Rect2d bb;

      bool fRet = tc.tracker->update(frame, bb);

      if (fRet)
      {
        if (IsRectInsideFrame(bb, frame))
        {
          tc.trail.push_back(bb);

          for(auto& b : tc.trail)
          {
            cv::circle(frame, b.br(), 1, cv::Scalar(255,0,0));
          }

          cv::rectangle(frame, bb, cv::Scalar(255, 0, 0 ), -1, 1);
        }
        else
        {
          Trackers.erase(Trackers.begin() + i);

          std::cout << "Tracker at " << i << " out of bound, size : " << Trackers.size() << "\n";
        }
        
      }
      else
      {
        Trackers.erase(Trackers.begin() + i);

        std::cout << "Tracker at " << i << " lost, size : " << Trackers.size() << "\n";
      }
    }

    cv::putText(frame, std::to_string(people), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("Motion Capture", frame);

    if (!ProcessKeyboard(cv::waitKey(1), cap, count)) break;
  }   
}
