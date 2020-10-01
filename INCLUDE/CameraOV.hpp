#ifndef CAMERAOV_HPP
#define CAMERAOV_HPP

#include <thread>
#include <chrono>
#include <functional>

using TOnCameraEventCbk = std::function<void (const std::string&, const std::string&, std::vector<uint8_t>&)>;

#include <CSubject.hpp>
#include <Encryption.hpp>

// face detection : age and gender
int fd_main(int argc, char *argv[]);
void fd_setcbk(TOnCameraEventCbk cbk);
void fd_stop(void);
// pedestrian tracker
int pt_main(int argc, char *argv[]);
void pt_setcbk(TOnCameraEventCbk cbk);
void pt_stop(void);
// face recognition
int fr_main(int argc, char *argv[]);
void fr_setcbk(TOnCameraEventCbk cbk);
void fr_stop(void);

class CCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CCamera(const std::string& source, const std::string& target)
    {
      iSource = source;
      iTarget = target;
    }

    ~CCamera()
    {
      Stop();
    }

    void Start(TOnCameraEventCbk cbk = nullptr)
    {
      if (iTarget == "face")
      {
        fd_setcbk(cbk);
      }
      else if (iTarget == "fr") 
      {
        fr_setcbk(cbk);
      }
      else if (iTarget == "person")
      {
        pt_setcbk(cbk);
      }

      iRunThread = std::thread(&CCamera::Run, this);
    }

    void Stop(void)
    {
      if (iTarget == "face")
      {
        fd_stop();  
      }
      else if (iTarget == "fr") 
      {
        fr_stop();
      }
      else if (iTarget == "person")
      {
        pt_stop();
      }

      if (iRunThread.joinable())
      {
        iRunThread.join();
      }

      std::cout << "camera thread joined\n";
    }

    void Play(void)
    {
      std::lock_guard<std::mutex> lg(iLock);
    }

    void Pause(void)
    {
      std::lock_guard<std::mutex> lg(iLock);
    }

    void StopPlay(void)
    {
      std::lock_guard<std::mutex> lg(iLock);
    }

    void Forward(void)
    {

    }

    void Backward(void)
    {

    }

    bool IsStarted()
    {
      std::lock_guard<std::mutex> lg(iLock);
      return iRunThread.joinable() ? true : false;
    }

    bool IsPaused()
    {
      std::lock_guard<std::mutex> lg(iLock);
      return false;
    }
 
    void Run(void)
    {
      if (iTarget == "face")
      {
        char *argv[] =
         {
           "demo",
           "-i", (char *)iSource.c_str(),
           "-m", "../cpp-cvl/MODELS/face-detection-adas-0001/FP16/face-detection-adas-0001.xml",
           "-m_ag", "../cpp-cvl/MODELS/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml"
         };

        fd_main(7, argv);
      }
      else if (iTarget == "fr")
      {
        char *argv[] =  {"a", "b"};
        fr_main(4, argv);
      }
      else if (iTarget == "person")
      {
        char *argv[] =
         {
           "demo",
           "-i", (char *)iSource.c_str(),
           "-m_det", "../cpp-cvl/MODELS/person-detection-retail-0013/FP16/person-detection-retail-0013.xml",
           "-m_reid", "../cpp-cvl/MODELS/person-reidentification-retail-0270/FP16/person-reidentification-retail-0270.xml"
         };

        pt_main(7, argv);
      }
      else 
      {
        std::cout << "Invalid target\n";
      }
    }

  protected:

    std::thread iRunThread;

    std::string iSource;
    
    std::string iTarget; 
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif