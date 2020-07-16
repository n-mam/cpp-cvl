#ifndef SOURCE_HPP
#define SOURCE_HPP 

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

class Source
{
  public:

    Source(const std::string& s)
    {
      iCapture = cv::VideoCapture(s.c_str()); 
    }
    
    Source(int c)
    {
      iCapture = cv::VideoCapture(c); 
    }

    ~Source()
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
      }

      return fRet;
    }

    bool HasEnded(void)
    {
      return (GetTotalFrames() == GetCurrentOffset());
    }

    bool HandleUserInput(void)
    {
      bool fRet = true;

      int c = cv::waitKey(1);

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
          auto o = iCurrentOffset + 5;

          if (o <= iCapture.get(cv::CAP_PROP_FRAME_COUNT))
          {
            iCapture.set(cv::CAP_PROP_POS_FRAMES, iCurrentOffset = o);
          }
        }
        else if (c == 'q' || c == 'Q')
        {
          fRet = false;
        }
      }

      return fRet;
    }

  protected:

    cv::VideoCapture iCapture;

    size_t iCurrentOffset = 0;

};

#endif