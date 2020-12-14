#ifndef CAMERACV_HPP
#define CAMERACV_HPP
#include <windows.h>

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

    CCamera(const std::string& source, const std::string& target, const std::string& algo)
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

        if (!iFRModel)
        {
          iFRModel = cv::face::LBPHFaceRecognizer::create(1,8,8,8,100);

          std::vector<cv::Mat> images;
          std::vector<int> labels;

          std::string basepath = "../../cpp-cvl/MODELS/";  //"./MODELS/" ;

          for (int i = 1; i <= 8; i++)
          {
            images.push_back(cv::imread(basepath + "o" + std::to_string(i) + ".jpg", 0));
            labels.push_back(0);
          }
          for (int i = 1; i <= 6; i++)
          {
            images.push_back(cv::imread(basepath + "b" + std::to_string(i) + ".jpg", 0));
            labels.push_back(1);
          }

          iFRModel->train(images, labels);
        }
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

    ~CCamera()
    {
      Stop();
    }

    void Start(TOnCameraEventCbk cbk = nullptr)
    {
      if (iSource->isOpened())
      {
        SetProperty("stop", "false");
        SetProperty("pause", "false");

        iOnCameraEventCbk = cbk;

        iTracker->AddEventListener(iDetector)->AddEventListener(shared_from_this());

        iRunThread = std::thread(&CCamera::Run, this);
      }
      else
      {
        std::cout << "Camera source not opened\n";
      }
    }

    void Stop(void)
    {
      SetProperty("stop", "true");

      if (IsStarted())
      {
        iRunThread.join();
      }

      iTracker->RemoveAllEventListeners();
      iDetector->RemoveAllEventListeners();
      RemoveAllEventListeners();
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

    virtual void OnEvent(std::any e)
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

      if (iFRModel)
      {
        UpdateFRModel(std::get<3>(in));
      }
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

    void Run(void)
    {
      cv::Mat frame;

      DWORD startTime = GetTickCount();

      while (!GetPropertyAsBool("stop"))
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

        iTracker->RenderDisplacementAndPaths(frame, GetName() == "CV");

        {
          std::lock_guard<std::mutex> lg(iLock);

          auto fps = (float) iSource->GetCurrentOffset() / (float)((GetTickCount() - startTime) / 1000);
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
          if (GetName() == "TESTCV")
          {
            cv::imshow(this->GetName().c_str(), frame);
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

    static int iNextLabel;

    static cv::Ptr<cv::face::LBPHFaceRecognizer> iFRModel;

    static void UpdateFRModel(TrackingContext& tc) 
    {
      if (tc.iThumbnails.size() < 20) return;

      std::vector<cv::Mat> gray;

      for (auto& m : tc.iThumbnails)
      {
        cv::Mat g;
        cv::cvtColor(m, g, cv::COLOR_BGR2GRAY);
        cv::resize(g, g, cv::Size(150, 150));
        gray.push_back(g);
      }

      int predictedLabel = -1, selectedLabel = -1;
      double confidence = 1000, selectedconfidence = 1000;

      for (auto& thumb : gray)
      {
        iFRModel->predict(thumb, predictedLabel, confidence);

        //std::cout << "predictedLabel : " << predictedLabel << ", confidence : " << confidence << "\n";

        if (confidence < selectedconfidence)
        {
          selectedconfidence = confidence;
          selectedLabel = predictedLabel;
        }
      }

      if (selectedconfidence > 95)
      {
        iFRModel->update(gray, std::vector<int>(gray.size(), iNextLabel));
        std::cout << "FR update, id : " << iNextLabel << "\n";
        iNextLabel++;
      }
      else
      {
        std::cout << "FR match, id : " << selectedLabel << " confidence : " << selectedconfidence << "\n";
      }
    }
};

int CCamera::iNextLabel = 2;

cv::Ptr<cv::face::LBPHFaceRecognizer> CCamera::iFRModel = nullptr;

using SPCCamera = std::shared_ptr<CCamera>;

#endif

// auto& f1 = std::async(std::launch::async, &CTracker::UpdateTrackingContexts, iTracker.get(), frame);
// auto& f2 = std::async(std::launch::async, &CDetector::Detect, iDetector.get(), frame);
// auto updates = f1.get(); 
// auto detections = f2.get();