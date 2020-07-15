#ifndef TRACKER_HPP
#define TRACKER_HPP 

bool IsRectInsideFrame(cv::Rect2d& r, cv::Mat& m)
{
  return ((static_cast<cv::Rect>(r) & cv::Rect(0, 0, m.cols, m.rows)) == static_cast<cv::Rect>(r));
}

bool ProcessKeyboard(int c, cv::VideoCapture cap, int count)
{
  bool fRet = true;

  if(c >= 0)
  {
    if (c == 'p' || c == 'P' || c == 0x20)
    {
      cv::waitKey(-1);
    }
    else if (c == 'r' || c == 'R')
    {
      auto o = count - 5;

      if (o >= 0)
      {
        cap.set(cv::CAP_PROP_POS_FRAMES, count = o);
      }
    }
    else if (c == 'f' || c == 'F')
    {
      auto o = count + 5;

      if (o <= cap.get(cv::CAP_PROP_FRAME_COUNT))
      {
        cap.set(cv::CAP_PROP_POS_FRAMES, count = o);
      }
    }
    else if (c == 'q' || c == 'Q')
    {
      fRet = false;
    }
  }

  return fRet;
}

#endif