#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

#include <Camera.hpp>
#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>

int up = 0;
int down = 0;
int left = 0;
int right = 0;

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  std::string source;

  if (argv[1])
  {
    source = argv[1];
  }
  else
  {
    source = "rtsp://neelabh.mam:welcome123@10.0.0.4:554/stream2";
  }

  Source s(source);

  if(!s.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  auto detector = ObjectDetector();
  auto tracker = OpenCVTracker();

  cv::Mat frame;

  for (;;)
  {
    if(!s.Read(frame))
	  {
      if(s.HasEnded())
      {
        s.Rewind();
        up = down = left = right = 0;
        tracker.ClearAllContexts();        
        continue;
      }
      else
      {
        break;
      }
    }

    if (frame.cols > 600)
    {
      auto scale = (float) 600 / frame.cols;
      cv::resize(frame, frame, cv::Size(0, 0), scale, scale);
    }

    if (frame.channels() == 4)
    {
      cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    cv::Mat temp = frame.clone();
    /*
     * update all active trackers first
     */
    tracker.UpdateTrackingContexts(temp,
      [&frame, &s](auto& tc)
      {
        auto start = GetRectCenter(tc.iTrail.front());
        auto end = GetRectCenter(tc.iTrail.back());

        auto [u, d, l, r] = s.IsPathIntersectingRefLine(start, end, frame);

        if (u || d || l || r)
        {
          up += u;
          down += d;
          left += l;
          right += r;
          return true;
        }

        return false;
      }
    );

    /*
     * run detector now using the updated (masked) frame 
     * That way, only new detections would be reported
     */
    if (s.GetCurrentOffset() % 4 == 0)
    {
      auto rects = detector.Detect(temp);

      for (auto& r : rects)
      {
        tracker.AddNewTrackingContext(temp, r);
      }
    }

    tracker.RenderTrackingContextsPath(frame);
    tracker.RenderTrackingContextsDisplacement(frame);

    cv::putText(frame, "u : " + std::to_string(up), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::putText(frame, "d : " + std::to_string(down), cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::putText(frame, "l : " + std::to_string(left), cv::Point(5, 70), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::putText(frame, "r : " + std::to_string(right), cv::Point(5, 90), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::line(frame, s.GetRefStartPoint(frame), s.GetRefEndPoint(frame), cv::Scalar(0, 0, 255), 1);

	  cv::imshow("People Counting", frame);

    if (!s.HandleUserInput()) break;
  }   
}
