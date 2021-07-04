#ifndef CAMERACV_HPP
#define CAMERACV_HPP

#include <thread>
#include <chrono>
#include <future>
#include <functional>

#include <Source.hpp>
#include <Tracker.hpp>
#include <Detector.hpp>
#include <Geometry.hpp>

#include <CSubject.hpp>

#include <opencv2/face/facerec.hpp>

class CCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CCamera() {}

    CCamera(const std::string& source, const std::string& target, const std::string& algo)
    {
      SetProperty("skipcount", "0");
      SetProperty("rtsp_transport", "tcp");
	    putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp");

      if (isdigit(source[0]))
      {
        iSource = std::make_shared<CSource>(std::stoi(source));
      }
      else
      {
        iSource = std::make_shared<CSource>(source);
      }

      if (0)
      {
        iDetector = std::make_shared<IEDetector>(target);
      }
      else if (target == "people")
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

      iTracker = std::make_shared<CTracker>();
    }

    virtual ~CCamera()
    {
      Stop();
    }

    virtual bool Start(TOnCameraEventCbk cbk = nullptr)
    {
      bool fRet = false;

      if (iSource->isOpened())
      {
        SetProperty("stop", "false");
        SetProperty("pause", "false");
        SetProperty("play", "false");

        iOnCameraEventCbk = cbk;

        iTracker->AddEventListener(iDetector)->AddEventListener(shared_from_this());

        iRunThread = std::thread(&CCamera::Run, this);

        fRet = true;
      }
      else
      {
        std::cout << "Camera source not opened\n";
      }

      return fRet;
    }

    virtual void Stop(void)
    {
      SetProperty("stop", "true");

      if (IsStarted())
      {
        iRunThread.join();
      }

      if (iTracker)
      {
        iTracker->RemoveAllEventListeners();
      }
      if (iDetector)
      {
        iDetector->RemoveAllEventListeners();
      }

      RemoveAllEventListeners();
    }

    virtual void Forward(void)
    {
      iSource->Forward();
    }

    virtual void Backward(void)
    {
      iSource->Backward();
    }

    virtual bool IsStarted()
    {
      std::lock_guard<std::mutex> lg(iLock);
      return iRunThread.joinable() ? true : false;
    }

    virtual void OnEvent(std::any e) override
    {
      auto in = std::any_cast<
        std::reference_wrapper<
          std::tuple<
            std::string,
            std::string,
            std::vector<uchar>,
            TrackingContext&
          >
        >>(e).get();

      iOnCameraEventCbk(
        "trail",
        std::get<0>(in), //path
        std::get<1>(in), //demography
        std::get<2>(in)  //thumbnail
      );
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

    std::string GetProperty(const std::string& key) override
    {
      if (key == "bbarea" ||
          key == "exhzbb")
      {
        return iDetector->GetProperty(key);
      }

      return CSubject<uint8_t, uint8_t>::GetProperty(key);
    }

    virtual void Run(void)
    {
      cv::Mat frame;

      auto started_at = std::chrono::high_resolution_clock::now();

      while (!GetPropertyAsBool("stop"))
      {
        if (!iSource->Read(frame))
        {
          if (iSource->HasEnded())
          {
            iSource->Rewind();
            iTracker->ClearAllContexts();
            started_at = std::chrono::high_resolution_clock::now();
            continue;
          }
          else
          {
            break;
          }
        }

        if (frame.cols > 400)
        {
          auto scale = (float) 400 / frame.cols;
          cv::resize(frame, frame, cv::Size(0, 0), scale, scale);
        }

        if (frame.channels() == 4)
        {
          cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        }

        auto skipcount = GetPropertyAsInt("skipcount");

        if ((iSource->GetCurrentOffset() % (skipcount + 1)) == 0)
        { /*
           * update all active trackers
           */
          auto updates = iTracker->UpdateTrackingContexts(frame);
          /*
           * run detector
           */
          auto detections = iDetector->Detect(frame);

          FilterDetections(detections, frame);
          /*
           * Match detections with the best tracking context
           */
          iTracker->MatchDetectionWithTrackingContext(detections, frame);
          /*
           * at this point every tracking context will potentially have 
           * a best match detection assigned. Add remaining detections to tracking
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

        iTracker->RenderDisplacementAndPaths(frame, GetProperty("name") == "CV");

        {
          std::lock_guard<std::mutex> lg(iLock);
          auto finished_at = std::chrono::high_resolution_clock::now();
          auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished_at - started_at).count();
          auto fps = (float) iSource->GetCurrentOffset() / (float)(duration_ms / 1000);
          cv::putText(frame, "FPS : " + std::to_string(fps), cv::Point(5, 10), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }

        if (GetPropertyAsBool("play"))
        {
          std::vector<uchar> buf;
          cv::imencode(".jpg", frame, buf);
          iOnCameraEventCbk("play", "", "", buf);
        }
        else
        {
          if (GetProperty("name") == "CV")
          {
            cv::imshow(this->GetProperty("name").c_str(), frame);
          }
        }

        if (!iSource->HandleUserInput(iTracker)) break;

        while (GetPropertyAsBool("pause") && !GetPropertyAsBool("stop"))
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

    std::thread iRunThread;

    SPCSource iSource;

    SPCTracker iTracker;

    SPCDetector iDetector;

    TOnCameraEventCbk iOnCameraEventCbk = nullptr;
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif

// auto& f1 = std::async(std::launch::async, &CTracker::UpdateTrackingContexts, iTracker.get(), frame);
// auto& f2 = std::async(std::launch::async, &CDetector::Detect, iDetector.get(), frame);
// auto updates = f1.get(); 
// auto detections = f2.get();