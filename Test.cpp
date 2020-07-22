#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <Camera.hpp>
#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>

int up = 0;
int down = 0;

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  if (!argv[1]) return 0;

  Source s(argv[1]);

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
        up = down = 0;
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

    cv::line(frame, cv::Point(0, frame.rows/2), cv::Point(frame.cols, frame.rows/2), cv::Scalar(0, 0, 255), 1);

    cv::Mat temp = frame.clone();
    /*
     * update all active trackers first
     */
    tracker.UpdateTrackingContexts(temp,
      [&frame](auto& tc)
      {
        bool fRet = false;

        auto start = GetRectCenter(tc.iTrail.front());
        auto end = GetRectCenter(tc.iTrail.back());

        auto refy = frame.rows / 2;

        if ((start.y < refy) && (end.y >= refy))
        {
          down++, fRet = true;
          std::cout << "start-y : " << start.y << " end-y : " << end.y << ", down\n";
        }
        else if ((start.y > refy) && (end.y <= refy))
        {
          up++, fRet = true;
          std::cout << "start-y : " << start.y << " end-y : " << end.y << ", up\n";          
        }

        return fRet;
      }
    );

    tracker.RenderTrackingContexts(frame);
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

    cv::putText(frame, "d : " + std::to_string(down), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
    cv::putText(frame, "u : " + std::to_string(up), cv::Point(5, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("People Counting", frame);

    if (!s.HandleUserInput()) break;
  }   
}
