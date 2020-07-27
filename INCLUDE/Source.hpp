#ifndef SOURCE_HPP
#define SOURCE_HPP 

#include <Counter.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

class CSource
{
  public:

    CSource(const std::string& s)
    {
      iCapture = cv::VideoCapture(s.c_str());
    }
    
    CSource(int c)
    {
      iCapture = cv::VideoCapture(iCamera = c); 
    }

    ~CSource()
    {
    }

    bool isOpened(void)
    {
      return iCapture.isOpened();
    }

    uint64_t GetTotalFrames(void)
    {
      return iCapture.get(cv::CAP_PROP_FRAME_COUNT);
    }

    uint64_t GetCurrentOffset(void)
    {
      return iCurrentOffset;
    }

    void Rewind(uint64_t offset = 0)
    {
      iCapture.set(cv::CAP_PROP_POS_FRAMES, offset);
      iCurrentOffset = 0;
    }

    bool Read(cv::Mat& m)
    {
      bool fRet = iCapture.read(m);

      if (fRet)
      {
        iCurrentOffset++;

        if (iCamera != -1)
        {
          cv::flip(m, m, 1);
        }
      }

      return fRet;
    }

    bool HasEnded(void)
    {
      return (GetTotalFrames() == GetCurrentOffset());
    }

    bool HandleUserInput(SPCCounter spcounter)
    {
      bool fRet = true;

      int c = cv::waitKey(25);

      if(c >= 0)
      {
        if (c == 'p' || c == 'P' || c == 0x20)
        {
          cv::waitKey(-1);
        }
        else if (c == 'r' || c == 'R')
        {
          auto o = iCurrentOffset - 5;

          if (o >= 0)
          {
            iCapture.set(cv::CAP_PROP_POS_FRAMES, iCurrentOffset = o);
          }
        }
        else if (c == 'f' || c == 'F')
        {
          auto offset = iCurrentOffset + 5;

          if (offset <= iCapture.get(cv::CAP_PROP_FRAME_COUNT))
          {
            iCapture.set(cv::CAP_PROP_POS_FRAMES, iCurrentOffset = offset);
          }
        }
        else if (c == 'a' || c == 'd')
        {
          if (c == 'a') spcounter->SetRefLine(0, -10);
          if (c == 'd') spcounter->SetRefLine(0, 10);
        }
        else if (c == 'w' || c == 's')
        {
          if (c == 'w') spcounter->SetRefLine(1, -10);
          if (c == 's') spcounter->SetRefLine(1, -10); 
        }
        else if (c == 'q' || c == 'Q')
        {
          fRet = false;
        }
      }

      return fRet;
    }
    
  protected:

    int iCamera = -1;

    std::string iSource;

    cv::VideoCapture iCapture;

    size_t iCurrentOffset = 0;
};

using SPCSource = std::shared_ptr<CSource>;

#endif