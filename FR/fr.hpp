#ifndef FRMAIN_HPP
#define FRMAIN_HPP

#include <string>
#include <vector>
#include <functional>
#include <inttypes.h>

#include <opencv2/opencv.hpp>

using TOnCameraEventCbk = std::function<
  void (const std::string&, const std::string&, const std::string&, std::vector<uint8_t>&)
>;

struct FR
{
  bool iStop = false;
  bool iPause = false;
  bool iPlay = true;

  TOnCameraEventCbk iOnCameraEventCbk = nullptr;

  void fr_stop(void)
  {
    iStop = true;
  }

  void fr_pause(bool bPause = true)
  {
    iPause = bPause;
  }

  void fr_play(bool bPlay = true)
  {
    iPlay = bPlay;
  }

  void fr_setcbk(TOnCameraEventCbk cbk)
  {
    iOnCameraEventCbk = cbk;
  }

  void ProcessFrame(const cv::Mat& frame)
  {
    auto clone = frame.clone();

    if (clone.cols > 600)
    {
      auto scale = (float) 600 / clone.cols;
      cv::resize(clone, clone, cv::Size(0, 0), scale, scale);
    }
    std::vector<uchar> buf;
    cv::imencode(".jpg", clone, buf);
    iOnCameraEventCbk("play", "", "", buf);
  }

  std::string iModelHomeDir;

  int fr_main(int argc, char* argv[], FR *);
};

#endif