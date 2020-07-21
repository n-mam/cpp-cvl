#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <Common.hpp>

struct TrackinContext
{
  cv::Ptr<cv::Tracker>      iTracker;  // cv tracker
  std::vector<cv::Rect2d>   iBBTrail;  // last bb
};

class TrackingManager
{
  public:

    TrackingManager()
    {
    }

    ~TrackingManager()
    {
      iTrackingContexts.clear();
    }

    size_t GetContextCount(void)
    {
      return iTrackingContexts.size();
    }

    void ClearAllContexts(void)
    {
      iTrackingContexts.clear();
    }

    void DisplayTrackingContexts(cv::Mat& m)
    {
      for (auto& tc : iTrackingContexts)
      {
        for (int i = 1; i < tc.iBBTrail.size() - 1; i++)
        {
          auto f = GetRectCenter(tc.iBBTrail[i]);
          auto b = GetRectCenter(tc.iBBTrail[i - 1]);
          cv::line(m, f, b, cv::Scalar(0, 255, 0), 1);
        }

        cv::rectangle(m, tc.iBBTrail.back(), cv::Scalar(255, 0, 0 ), 1, 1);
      }
    }

    virtual void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) {}

    virtual void UpdateTrackingContexts(const cv::Mat& frame) {}

  protected:

    std::vector<TrackinContext> iTrackingContexts;
};

class OpenCVTracker : public TrackingManager
{
  public:

    void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) override
    {
      TrackinContext tc;

      tc.iTracker = cv::TrackerCSRT::create();

      tc.iBBTrail.push_back(r);

      tc.iTracker->init(m, r);

      iTrackingContexts.push_back(tc);
    }

    void UpdateTrackingContexts(const cv::Mat& m) override
    {
      if (!iTrackingContexts.size()) return;

      for (int i = (iTrackingContexts.size() - 1); i >= 0; i--)
      {
        auto& tc = iTrackingContexts[i];

        cv::Rect2d bb;

        bool fRet = tc.iTracker->update(m, bb);

        if (fRet)
        {
          if (IsRectInsideMat(bb, m))
          {
            m(bb) = 1;
            tc.iBBTrail.push_back(bb);
          }
          else
          {
            iTrackingContexts.erase(iTrackingContexts.begin() + i);
            std::cout << "Tracker at " << i << " out of bound, size : " << iTrackingContexts.size() << "\n";
          }
        }
        else
        {
          iTrackingContexts.erase(iTrackingContexts.begin() + i);

          std::cout << "Tracker at " << i << " lost, size : " << iTrackingContexts.size() << "\n";
        }
      }
    }

  protected:

};

#endif //TRACKER_HPP