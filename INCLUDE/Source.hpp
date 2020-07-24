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
      iCapture = cv::VideoCapture(iCamera = c);
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
          auto offset = iCurrentOffset + 5;

          if (offset <= iCapture.get(cv::CAP_PROP_FRAME_COUNT))
          {
            iCapture.set(cv::CAP_PROP_POS_FRAMES, iCurrentOffset = offset);
          }
        }
        else if (c == 'a' || c == 'd')
        {
          iRefOrientation = 0;
          if (c == 'a') iRefDelta -= 10;
          if (c == 'd') iRefDelta += 10; 
        }
        else if (c == 'w' || c == 's')
        {
          iRefOrientation = 1;
          if (c == 'w') iRefDelta -= 10;
          if (c == 's') iRefDelta += 10; 
        }
        else if (c == 'q' || c == 'Q')
        {
          fRet = false;
        }
      }

      return fRet;
    }

    cv::Point GetRefStartPoint(cv::Mat& m)
    {
      if (iRefOrientation)
      {/*
        * horizontal
        */
        return cv::Point(0, m.rows/2 + iRefDelta);
      }
      else
      {
        return cv::Point(m.cols/2 + iRefDelta, 0);
      }
    }

    cv::Point GetRefEndPoint(cv::Mat& m)
    {
      if (iRefOrientation)
      {/*
        * horizontal
        */
        return cv::Point(m.cols, m.rows/2 + iRefDelta);
      }
      else
      {
        return cv::Point(m.cols/2 + iRefDelta, m.rows);
      }
    }

    std::tuple<int, int, int, int> IsPathIntersectingRefLine(cv::Point start, cv::Point end, cv::Mat& m)
    {
      int u = 0, d = 0, l = 0, r = 0;

      if (iRefOrientation)
      {
        auto refy = m.rows/2 + iRefDelta;

        if ((start.y < refy) && (end.y >= refy))
        {
          d++;
          std::cout << "start-y : " << start.y << " end-y : " << end.y << ", down\n";
        }
        else if ((start.y > refy) && (end.y <= refy))
        {
          u++;
          std::cout << "start-y : " << start.y << " end-y : " << end.y << ", up\n";          
        }
      }
      else
      {
        auto refx = m.cols/2 + iRefDelta;

        if ((start.x < refx) && (end.x >= refx))
        {
          r++;
          std::cout << "start-x : " << start.x << " end-x : " << end.x << ", right\n";
        }
        else if ((start.x > refx) && (end.x <= refx))
        {
          l++;
          std::cout << "start-x : " << start.x << " end-x : " << end.x << ", left\n";          
        }
      }

      return std::make_tuple(u, d, l, r);
    }

    
  protected:

    int iCamera = -1;

    std::string iSource;

    cv::VideoCapture iCapture;

    size_t iCurrentOffset = 0;

    char iRefOrientation = 1; // horizontal

    int iRefDelta = 0; // ref line offset from baseline
};

#endif