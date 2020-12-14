#ifndef CAMERAOV_HPP
#define CAMERAOV_HPP

#include <thread>
#include <chrono>
#include <functional>

#include <Geometry.hpp>
#include <CSubject.hpp>
#include <Encryption.hpp>

// face recognition
int fr_main(int argc, char *argv[]);
void fr_setcbk(TOnCameraEventCbk cbk);
void fr_play(bool);
void fr_pause(bool);
void fr_stop(void);

class COVCamera : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    COVCamera(const std::string& source, const std::string& target)
    {
      iSource = source;
      iTarget = target;
    }

    ~COVCamera()
    {
      Stop();
    }

    void Start(TOnCameraEventCbk cbk = nullptr)
    {
      if (iTarget == "fr") 
      {
        fr_setcbk(cbk);
      }

      iRunThread = std::thread(&COVCamera::Run, this);
    }

    void Stop(void)
    {
      if (iTarget == "fr") 
      {
        fr_stop();
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

      if (iTarget == "fr") 
      {
        fr_play(true);
        fr_pause(false);
      }
    }

    void Pause(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        fr_pause(true); 
      }
    }

    void StopPlay(void)
    {
      std::lock_guard<std::mutex> lg(iLock);

      if (iTarget == "fr") 
      {
        fr_play(false);
      }
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
      if (iTarget == "fr")
      {
        char *argv[] =  {"a", "b"};
        fr_main(4, argv);
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

using SPCOVCamera = std::shared_ptr<COVCamera>;

#endif