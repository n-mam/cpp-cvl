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

  bool IsFrozen(void)
  {
    bool fRet = false;

    if (iTrail.size() >= 8)
    {
      auto& last = iTrail[iTrail.size() - 1];
      auto& prev = iTrail[iTrail.size() - 8];

      auto d = Distance(GetRectCenter(last), GetRectCenter(prev));

      if (d <= 2) fRet = true;
    }

    return fRet;
  }
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

    bool DoesROIOverlapAnyContext(cv::Rect2d roi)
    {
      for (auto& tc : iTrackingContexts)
      {
        if (DoesRectOverlapRect(roi, tc.iTrail.back()))
        {
          return true;
        }
      }
      return false;
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

    void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& roi) override
    {
      TrackingContext tc;

      cv::TrackerCSRT::Params params;
      params.psr_threshold = 0.04f;

      tc.iTracker = cv::TrackerCSRT::create(params);

      tc.iTrail.push_back(roi);

      if (roi.x + roi.width > m.cols) return; //roi.width -= m.cols - roi.x;
      if (roi.y + roi.height > m.rows) return; //roi.height -= m.rows - roi.y;

      tc.iTracker->init(m, roi);

      iTrackingContexts.push_back(tc);
    }

    void UpdateTrackingContexts(const cv::Mat& m, TCbkTracker cbk = nullptr) override
    {
      if (!iTrackingContexts.size()) return;

      for (int i = (iTrackingContexts.size() - 1); i >= 0; i--)
      {
        auto& tc = iTrackingContexts[i];

        if (tc.IsFrozen())
        {
          iPurgedContexts.push_back(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + i);          
          std::cout << "removed frozen tc\n";
          continue;
        }

        cv::Rect2d bb;

        bool fRet = tc.iTracker->update(m, bb);

        if (fRet)
        {
          if (IsRectInsideMat(bb, m))
          {
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