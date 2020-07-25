#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <thread>

#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>
#include <Counter.hpp>
#include <Geometry.hpp>

#include <CSubject.hpp>

class CCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CCamera(const std::string& source)
    {
      iSource = std::make_shared<CSource>(source);
      iTracker = std::make_shared<OpenCVTracker>();
      iDetector = std::make_shared<ObjectDetector>();
      iCounter = std::make_shared<CCounter>();
    }

    ~CCamera()
    {
      Stop();
    }

    bool Start(void)
    {
      if(!iSource->isOpened())
      {
        std::cout << "Error opening video stream\n";
        return false;
      }

      iRunThread = std::thread(&CCamera::Run, this);

      return true;
    }

    void Stop()
    {
      iStop = true;

      if (iRunThread.joinable())
      {
        iRunThread.join();
      }
    }

    void Run(void)
    {
      cv::Mat frame;

      while (!iStop)
      {
        if (!iSource->Read(frame))
	      {
          if (iSource->HasEnded())
          {
            iSource->Rewind();
            iCounter.reset(new CCounter());
            iTracker->ClearAllContexts();        
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
        iTracker->UpdateTrackingContexts(temp,
          [this, &frame](auto& tc)
          {
            return iCounter->ProcessTrail(tc.iTrail, frame);
          });

        /*
         * run the detector and filter out 
         * tracker updated overlapping rois
         */
        auto detections = iDetector->Detect(temp);

        for (int i = (detections.size() - 1); detections.size() && i >= 0; i--)
        {
          if (iTracker->DoesROIOverlapAnyContext(detections[i]))
          {
            detections.erase(detections.begin() + i);       
          }
        }

        for (auto& roi : detections)
        {
          iTracker->AddNewTrackingContext(temp, roi);
        }

        iTracker->RenderTrackingContextsPath(frame);
        iTracker->RenderTrackingContextsDisplacement(frame);

        iCounter->DisplayCounts(frame);

        cv::line(frame, iCounter->GetRefStartPoint(frame), iCounter->GetRefEndPoint(frame), cv::Scalar(0, 0, 255), 1);

	      cv::imshow("People Counting", frame);

        if (!iSource->HandleUserInput(iCounter)) break;
      }
    }

  protected:

    SPCSource   iSource;
    SPCTracker  iTracker;
    SPCCounter  iCounter;
    SPCDetector iDetector;    
    std::thread iRunThread;
    bool iStop = false;
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif