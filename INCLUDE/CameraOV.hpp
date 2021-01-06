#ifndef CAMERAOV_HPP
#define CAMERAOV_HPP

#include <thread>
#include <chrono>
#include <functional>

#include <CameraCV.hpp>

// face recognition
#include "../FR/fr.hpp"

class COVCamera : public CCamera
{
  public:

    COVCamera(const std::string& source, const std::string& target)
    {
      iSource = source;
      iTarget = target;
    }

    virtual ~COVCamera()
    {
      Stop();
    }

    virtual void Start(TOnCameraEventCbk cbk = nullptr)
    {
      if (iTarget == "fr") 
      {
        iFR.fr_setcbk(cbk);
      }

      iRunThread = std::thread(&COVCamera::Run, this);
    }

    virtual void Stop(void)
    {
      if (iTarget == "fr") 
      {
        iFR.fr_stop();
      }
    }

    virtual void Play(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        iFR.fr_play(true);
        iFR.fr_pause(false);
      }
    }

    virtual void Pause(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        iFR.fr_pause(true); 
      }
    }

    virtual void StopPlay(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        iFR.fr_play(false);
      }
    }

    virtual void Forward(void) {}

    virtual void Backward(void) {}
 
    virtual void Run(void)
    {
      if (iTarget == "fr")
      {
        char * argv[] =  {
         "smart_classroom_demo.exe",
         "-m_fd", (char *)(GetModelHomeDir() + "face-detection-adas-0001/FP16/face-detection-adas-0001.xml"s).c_str(),
         "-m_lm", (char *)(GetModelHomeDir() + "landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml"s).c_str(),
         "-m_reid", (char *)(GetModelHomeDir() + "face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml"s).c_str(),
         "-fg", (char *)(GetModelHomeDir() + "fr_gallery/faces_gallery.json"s).c_str(),
         "-i", (char *)(iSource.c_str())
        };

        iFR.fr_main(sizeof(argv)/sizeof(char *), argv, &iFR);
      }
      else 
      {
        std::cout << "Invalid target\n";
      }
    }

  protected:

    FR iFR;

    std::string iSource;
    
    std::string iTarget; 
};

using SPCOVCamera = std::shared_ptr<COVCamera>;

#endif