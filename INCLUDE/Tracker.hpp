#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <Common.hpp>

struct TrackingContext
{
  cv::Ptr<cv::Tracker>      iTracker;  // cv tracker
  std::vector<cv::Rect2d>   iTrail;    // last bb
  bool iSkip = false;
};

using TCbkTracker = std::function<bool (const TrackingContext&)>;

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

    void RenderTrackingContextsDisplacement(cv::Mat& m)
    {
      for (auto& tc : iTrackingContexts)
      {
        auto first = GetRectCenter(tc.iTrail.front());
        auto last = GetRectCenter(tc.iTrail.back());
        cv::line(m, first, last, cv::Scalar(0, 0, 255), 1);
      }
      for (auto& pc : iPurgedContexts)
      {
        auto first = GetRectCenter(pc.iTrail.front());
        auto last = GetRectCenter(pc.iTrail.back());
        cv::line(m, first, last, cv::Scalar(0, 255, 255), 1);
      }
    }

    void RenderTrackingContextsPath(cv::Mat& m)
    {
      for (auto& tc : iTrackingContexts)
      {
        for (int i = 1; i < tc.iTrail.size() - 1; i++)
        {
          auto f = GetRectCenter(tc.iTrail[i]);
          auto b = GetRectCenter(tc.iTrail[i-1]);
          cv::line(m, f, b, cv::Scalar(0, 255, 0), 1);
        }

        cv::rectangle(m, tc.iTrail.back(), cv::Scalar(255, 0, 0 ), 1, 1);
      }
    }

    virtual void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) {}

    virtual void UpdateTrackingContexts(const cv::Mat& frame, TCbkTracker cbk = nullptr) {}

  protected:

    std::vector<TrackingContext> iTrackingContexts;
    std::vector<TrackingContext> iPurgedContexts;
};

class OpenCVTracker : public TrackingManager
{
  public:

    void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) override
    {
      TrackingContext tc;

      tc.iTracker = cv::TrackerCSRT::create();

      tc.iTrail.push_back(r);

      tc.iTracker->init(m, r);

      iTrackingContexts.push_back(tc);
    }

    void UpdateTrackingContexts(const cv::Mat& m, TCbkTracker cbk = nullptr) override
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

            tc.iTrail.push_back(bb);

            if (cbk && !tc.iSkip)
            {
              tc.iSkip = cbk(tc);
            }
          }
          else
          {
            iPurgedContexts.push_back(tc);
            iTrackingContexts.erase(iTrackingContexts.begin() + i);
            std::cout << "Tracker at " << i << " is out of the bounds, size : " << iTrackingContexts.size() << "\n";
          }
        }
        else
        {
          iPurgedContexts.push_back(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + i);
          std::cout << "Tracker at " << i << " lost, size : " << iTrackingContexts.size() << "\n";
        }
      }
    }

  protected:

};

#endif //TRACKER_HPP