#ifndef CAMERAOV_HPP
#define CAMERAOV_HPP

#include <thread>
#include <chrono>
#include <functional>

#include <CameraCV.hpp>

// face recognition
int fr_main(int argc, char *argv[]);
void fr_setcbk(TOnCameraEventCbk cbk);
void fr_play(bool);
void fr_pause(bool);
void fr_stop(void);

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
        fr_setcbk(cbk);
      }

      iRunThread = std::thread(&COVCamera::Run, this);
    }

    virtual void Stop(void)
    {
      if (iTarget == "fr") 
      {
        fr_stop();
      }
    }

    virtual void Play(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        fr_play(true);
        fr_pause(false);
      }
    }

    virtual void Pause(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        fr_pause(true); 
      }
    }

    virtual void StopPlay(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        fr_play(false);
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
         "-m_fd", "../../cpp-cvl/MODELS/face-detection-adas-0001/FP16/face-detection-adas-0001.xml",
         "-m_lm", "../../cpp-cvl/MODELS/landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml",
         "-m_reid", "../../cpp-cvl/MODELS/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml",
         "-fg", "F:/openvino/inference_engine/demos/smart_classroom_demo/faces_gallery.json",
         "-i", (char *)(iSource.c_str())
        };

        fr_main(sizeof(argv)/sizeof(char *), argv);
      }
      else 
      {
        std::cout << "Invalid target\n";
      }
    }

  protected:

    std::string iSource;
    
    std::string iTarget; 
};

using SPCOVCamera = std::shared_ptr<COVCamera>;

#endif