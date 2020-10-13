#ifndef CAMERACV_HPP
#define CAMERACV_HPP
#include <windows.h>

#include <thread>
#include <chrono>
#include <future>
#include <functional>

using TOnCameraEventCbk = std::function<void (const std::string&, const std::string&, const std::string&, std::vector<uint8_t>&)>;

#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>
#include <Geometry.hpp>

#include <CSubject.hpp>

class CCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CCamera(const std::string& source, const std::string& target, const std::string& algo, const std::string& tracker)
    {
      SetProperty("skipcount", "0");

      SetProperty("rtsp_transport", "udp");
	    putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;udp");

      if (isdigit(source[0]))
      {
        iSource = std::make_shared<CSource>(std::stoi(source));
      }
      else
      {
        iSource = std::make_shared<CSource>(source);
      }

      if (target == "people")
      {
        iDetector = std::make_shared<PeopleDetector>();
      }
      else if (target == "face")
      {
        iDetector = std::make_shared<FaceDetector>();
      }
      else if (target == "mocap")
      {
        iDetector = std::make_shared<BackgroundSubtractor>(algo);
      }
      else
      {
        iDetector = std::make_shared<ObjectDetector>(target);
      }

      iTracker = std::make_shared<OpenCVTracker>(tracker);
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

    virtual void SetProperty(const std::string& key, const std::string& value) override
    {
      if (key == "transport")
      {
        if (value == "tcp")
        {
          putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp");
        }
        else if (value == "udp")
        {
          putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;udp");
        }
        else
        {
          return;
        }
      }
      else if (key == "bbarea" ||
               key == "exhzbb")
      {
        iDetector->SetProperty(key, value);
        return;
      }

      CSubject<uint8_t, uint8_t>::SetProperty(key, value);
    }

    std::string GetProperty(const std::string& key)
    {
      if (key == "bbarea" ||
          key == "exhzbb")
      {
        return iDetector->GetProperty(key);
      }

      return CSubject<uint8_t, uint8_t>::GetProperty(key);
    }

    void Run(void)
    {
      cv::Mat frame;

      DWORD startTime = GetTickCount();

      while (!iStop)
      {
        if (!iSource->Read(frame))
	      {
          if (iSource->HasEnded())
          {
            iSource->Rewind();
            iTracker->ClearAllContexts();
            startTime = GetTickCount();
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

        auto skipcount = GetPropertyAsInt("skipcount");

        if ((iSource->GetCurrentOffset() % (skipcount + 1)) == 0)
        { /*
           * update all active trackers first
           */
          auto updates = iTracker->UpdateTrackingContexts(frame);
          /*
           * run detector
           */
          auto detections = iDetector->Detect(frame);
          /*
           * Match detections with the best tracking context
           */
          iTracker->MatchDetectionWithTrackingContext(detections, frame);
          /*
           * at this point every tracking context will potentially have a best match detection assigned
           */
          for (auto& d : detections)
          {
            if (!std::get<3>(d))
            {
              auto tc = iTracker->AddNewTrackingContext(frame, std::get<0>(d));

              if (tc)
              {
                tc->updateAge(std::get<1>(d));
                tc->updateGender(std::get<2>(d));
              }
            }
          }

        }

        iTracker->RenderDisplacementAndPaths(frame);

        {
          std::lock_guard<std::mutex> lg(iLock);

          auto fps = (float) iSource->GetCurrentOffset() / (float)((GetTickCount() - startTime) / 1000);
          cv::putText(frame, "FPS : " + std::to_string(fps), cv::Point(5, 10), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

          if (iPlay)
          {
            std::vector<uchar> buf;
            cv::imencode(".jpg", frame, buf);
            iOnCameraEventCbk("play", "", "", buf);
          }
          else
          {
            if (GetName() == "CV")
            {
              cv::imshow(this->GetName().c_str(), frame);
            }
          }
        }

        if (!iSource->HandleUserInput(iTracker)) break;

        while (iPaused && !iStop)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
      }

      if (iOnCameraEventCbk)
      {
        iOnCameraEventCbk("stop", "", "", std::vector<uint8_t>());
      }      
    }

  protected:

    bool iStop = false;

    bool iPlay = false;

    bool iPaused = false;

    std::thread iRunThread;

    SPCSource   iSource;

    SPCTracker  iTracker;

    SPCDetector iDetector;

    TOnCameraEventCbk iOnCameraEventCbk = nullptr;    
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif

// auto& f1 = std::async(std::launch::async, &CTracker::UpdateTrackingContexts, iTracker.get(), frame);
// auto& f2 = std::async(std::launch::async, &CDetector::Detect, iDetector.get(), frame);
// auto updates = f1.get(); 
// auto detections = f2.get();