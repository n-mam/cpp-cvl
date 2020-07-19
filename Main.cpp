#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>

uint64_t people = 0;

TrackingManager tm;

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  Source s("f:/tv2.mp4");

  if(!s.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  auto detector = ObjectDetector();

  cv::Mat frame;

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

    if (frame.cols > 600)
    {
      auto scale = (float) 600 / frame.cols;

      cv::resize(frame, frame, cv::Size(0, 0), scale, scale);
    }

    if (frame.channels() == 4)
    {
      cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    /*
     * update all active trackers first
     */
    tm.Update(frame);
    /*
     * Now detect all faces using the updated (masked) 
     * frame. That way, only new detections would happen
     */
    if (s.GetCurrentOffset() % 4 == 0)
    {
      auto rects = detector.Detect(frame);

      for (auto& r : rects)
      {
        tm.AddNewTrackingContext(frame, r);
      }
    }

    cv::putText(frame, std::to_string(people), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("People Counting", frame);

    if (!s.HandleUserInput()) break;
  }   
}
