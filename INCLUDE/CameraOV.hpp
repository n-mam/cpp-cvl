#ifndef CAMERAOV_HPP
#define CAMERAOV_HPP

#include <thread>
#include <chrono>
#include <functional>

// face detection : age and gender
int fd_main(int argc, char *argv[]); 
// pedestrian tracker
int pt_main(int argc, char *argv[]);
// face recognition
int fr_main(int argc, char *argv[]);

using TOnCameraEventCbk = std::function<void (const std::string&, const std::string&)>;

#include <CSubject.hpp>
#include <Encryption.hpp>

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
      iOnCameraEventCbk = cbk;

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
      return iPaused;
    }
    void Run(void)
    {
      char *argv[] =  {"a", "b"};

      if (iTarget == "fd")
      {
        fd_main(4, argv);
      }
      else if (iTarget == "fr")
      {
        fr_main(4, argv);
      }
      else if (iTarget == "pt")
      {
        pt_main(4, argv);
      }
      else 
      {
        std::cout << "Invalid target\n";
      }
    }

  protected:

    bool iStop = false;

    bool iPlay = false;

    bool iPaused = false;

    std::thread iRunThread;

    std::string iSource;
    
    std::string iTarget;

    TOnCameraEventCbk iOnCameraEventCbk = nullptr;    
};

using SPCCamera = std::shared_ptr<CCamera>;

#endif