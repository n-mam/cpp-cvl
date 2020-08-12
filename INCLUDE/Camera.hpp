#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <thread>
#include <chrono>
#include <functional>

using TOnCameraEventCbk = std::function<void (const std::string&, const std::string&)>;

#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>
#include <Counter.hpp>
#include <Geometry.hpp>

#include <CSubject.hpp>
#include <Encryption.hpp>

class CCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CCamera(const std::string& source, const std::string& target, const std::string& tracker)
    {
      if (isalpha(source[0]))
      {
        iSource = std::make_shared<CSource>(source);
      }
      else
      {
        iSource = std::make_shared<CSource>(std::stoi(source));
      }

      iTracker = std::make_shared<OpenCVTracker>(tracker);

      if (target == "person" || target == "car")
      {
        iDetector = std::make_shared<ObjectDetector>(target);
      }
      else if (target == "face")
      {
        iDetector = std::make_shared<FaceDetector>();
      }
      else if (target == "mocap")
      {
        iDetector = std::make_shared<BackgroundSubtractor>();
      }

      iCounter = std::make_shared<CCounter>();
    }

    ~CCamera()
    {
      Stop();
    }

    void Start(TOnCameraEventCbk cbk = nullptr)
    {
      if (!iSource->isOpened())
      {
        std::cout << "Error opening video stream\n";
        return;
      }

      iStop = false;

	    iPaused = false;

      iOnCameraEventCbk = cbk;

      iTracker->SetEventCallback(cbk);

      iRunThread = std::thread(&CCamera::Run, this);
    }

    void Stop(void)
    {
      iStop = true;

      if (iRunThread.joinable())
      {
        iRunThread.join();
      }
    }

    void Play(void)
    {
      std::lock_guard<std::mutex> lg(iLock);
      iPlay = true;
      iPaused = false;
    }

    void Pause(void)
    {
      std::lock_guard<std::mutex> lg(iLock);
      iPaused = true;
    }

    void StopPlay(void)
    {
      std::lock_guard<std::mutex> lg(iLock);
      iPlay = false;
    }

    void Forward(void)
    {
      iSource->Forward();
    }

    void Backward(void)
    {
      iSource->Backward();
    }

    bool IsStarted()
    {
      std::lock_guard<std::mutex> lg(iLock);
      return iRunThread.joinable() ? true : false;
    }

    bool IsPaused()
    {
      std::lock_guard<std::mutex> lg(iLock);
      return iPaused;
    }

    void SetSkipCount(int count = 0)
    {
      iSkipCount = count;
    }
    
    int GetSkipCount()
    {
      return iSkipCount;
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

        if (iSource->GetCurrentOffset() % GetSkipCount() == 0)
        {
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
            if (iTracker->DoesROIOverlapAnyContext(detections[i], frame))
            {
              detections.erase(detections.begin() + i);
            }
          }

          for (auto& roi : detections)
          {
            iTracker->AddNewTrackingContext(temp, roi);
          }
        }

        iTracker->RenderTrackingContextsPath(frame);
        iTracker->RenderTrackingContextsDisplacement(frame);

        iCounter->DisplayCounts(frame);

        cv::line(frame, iCounter->GetRefStartPoint(frame), iCounter->GetRefEndPoint(frame), cv::Scalar(0, 0, 255), 1);

        {
          std::lock_guard<std::mutex> lg(iLock);

          if (iPlay)
          {
            std::vector<uchar> buf;
            cv::imencode(".jpg", frame, buf);
            char encoded[360*500];
            int n = Base64Encode((unsigned char *)encoded, buf.data(), buf.size());
            iOnCameraEventCbk("play", std::string(encoded, n));
          }
          else
          {
            if (GetName() == "CV")
            {
              cv::imshow(this->GetName().c_str(), frame);
            }
          }
        }

        if (!iSource->HandleUserInput(iCounter)) break;

        while (iPaused && !iStop)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
      }

      if (iOnCameraEventCbk)
      {
        iOnCameraEventCbk("stop", "");
      }      
    }

  protected:

    bool iStop = false;

    bool iPlay = false;

    bool iPaused = false;

    int iSkipCount = 1;

    TOnCameraEventCbk iOnCameraEventCbk = nullptr;

    std::thread iRunThread;

    SPCSource   iSource;

    SPCTracker  iTracker;

    SPCCounter  iCounter;

    SPCDetector iDetector;
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif