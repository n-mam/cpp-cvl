#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <thread>
#include <chrono>

#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>
#include <Counter.hpp>
#include <Geometry.hpp>

#include <CSubject.hpp>

using TOnStopCbk = std::function<void (void)>;
using TPlayCbk = std::function<void (const std::string& frame)>;

class CCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CCamera(const std::string& source, const std::string& target, const std::string& tracker)
    {
      iSource = std::make_shared<CSource>(source);
      iTracker = std::make_shared<OpenCVTracker>(tracker);

      if (target == "person" || target == "car")
      {
        iDetector = std::make_shared<ObjectDetector>(target);
      }
      else if (target == "face")
      {
        iDetector = std::make_shared<FaceDetector>();
      }

      iCounter = std::make_shared<CCounter>();
    }

    ~CCamera()
    {
      Stop();
    }

    bool Start(TOnStopCbk cbk = nullptr)
    {
      if(!iSource->isOpened())
      {
        std::cout << "Error opening video stream\n";
        return false;
      }

      iOnStopCbk = cbk;

      iRunThread = std::thread(&CCamera::Run, this);

      return true;
    }

    void Stop(TOnStopCbk cbk = nullptr)
    {
      iStop = true;

      if (iRunThread.joinable())
      {
        iRunThread.join();
      }
    }

    void Play(TPlayCbk cbk)
    {
      iPlayCbk = cbk;
      iPause = false;
    }

    void Pause()
    {
      iPlayCbk = nullptr;
      iPause = true;
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

        if (iPlayCbk)
        {

        }
        else
        {
          cv::imshow(this->GetName().c_str(), frame);
        }

        while (iPause)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        if (!iSource->HandleUserInput(iCounter)) break;
      }

      if (iOnStopCbk)
      {
        iOnStopCbk();
      }      
    }

  protected:

    bool iStop = false;

    bool iPause = false;

    TOnStopCbk iOnStopCbk = nullptr;

    TPlayCbk iPlayCbk = nullptr;

    std::thread iRunThread;

    SPCSource   iSource;

    SPCTracker  iTracker;

    SPCCounter  iCounter;

    SPCDetector iDetector;
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif