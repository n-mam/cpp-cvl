#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <Counter.hpp>
#include <Geometry.hpp>

struct TrackingContext
{
  cv::Ptr<cv::Tracker> iTracker;   // cv tracker

  std::vector<cv::Rect2d> iTrail;  // track bb trail

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

class CTracker
{
  public:

    CTracker(const std::string& tracker)
    {
      iType = tracker;
      iCounter = std::make_shared<CCounter>();
    }

    ~CTracker()
    {
      ClearAllContexts();
    }

    void ClearAllContexts(void)
    {
      iTrackingContexts.clear();
      iCounter.reset(new CCounter());
    }

    void SetRefLine(int orientation, int delta)
    {
      iCounter->SetRefLine(orientation, delta);
    }

    size_t GetContextCount(void)
    {
      return iTrackingContexts.size();
    }

    void RenderDisplacementAndPaths(cv::Mat& m)
    {
      for (auto& tc : iTrackingContexts)
      { /*
         * Displacement
         */
        auto first = GetRectCenter(tc.iTrail.front());
        auto last = GetRectCenter(tc.iTrail.back());
        cv::line(m, first, last, cv::Scalar(0, 0, 255), 1);
        /*
         * Path
         */
        for (size_t i = 1; i < tc.iTrail.size() - 1; i++)
        {
          auto f = GetRectCenter(tc.iTrail[i]);
          auto b = GetRectCenter(tc.iTrail[i-1]);
          cv::line(m, f, b, cv::Scalar(0, 255, 0), 1);
        }

        auto& bb = tc.iTrail.back();
        cv::rectangle(m, bb, cv::Scalar(0, 0, 255), 1, 1); //tracking red
      }

      for (auto& pc : iPurgedContexts)
      {
        auto first = GetRectCenter(pc.iTrail.front());
        auto last = GetRectCenter(pc.iTrail.back());
        cv::line(m, first, last, cv::Scalar(0, 255, 255), 1);
      }

      iCounter->DisplayRefLineAndCounts(m);
    }

    bool DoesROIOverlapAnyContext(cv::Rect2d roi, cv::Mat& m)
    {
      for (auto& tc : iTrackingContexts)
      {
        if (DoesRectOverlapRect(roi, tc.iTrail.back()))
        {
          cv::rectangle(m, roi, cv::Scalar(255, 0, 0 ), 1, 1);  // detection blue
          return true;
        }
      }
      return false;
    }

    virtual void SetEventCallback(TOnCameraEventCbk cbk)
    {
      iOnCameraEventCbk = cbk;
    }

    virtual void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) {}

    virtual std::vector<cv::Rect2d> UpdateTrackingContexts(cv::Mat& frame, TCbkTracker cbk = nullptr) { return {}; }

  protected:
  
    std::string iType;

    SPCCounter iCounter;

    TOnCameraEventCbk iOnCameraEventCbk = nullptr;

    std::vector<TrackingContext> iTrackingContexts;

    std::vector<TrackingContext> iPurgedContexts;

    virtual void PurgeAndSaveTrackingContext(TrackingContext& tc)
    {
      if (iOnCameraEventCbk)
      {
        std::string data = "";

        for (auto& r : tc.iTrail)
        {
          auto p = GetRectCenter(r);

          if (data.size())
          {
            data += ", ";
          }

          data += std::to_string(p.x) + " " + std::to_string(p.y);
        }

        iOnCameraEventCbk("trail", data, std::vector<uint8_t>());
      }

      if (iPurgedContexts.size() > 20)
      {
        iPurgedContexts.clear();
      }

      iPurgedContexts.push_back(tc);
    }
};

class OpenCVTracker : public CTracker
{
  public:

    OpenCVTracker(const std::string& tracker) : CTracker(tracker) {}

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

    std::vector<cv::Rect2d> UpdateTrackingContexts(cv::Mat& m, TCbkTracker cbk = nullptr) override
    {
      if (!iTrackingContexts.size())
      {
        return {};
      } 

      std::vector<cv::Rect2d> out;

      for (size_t i = iTrackingContexts.size(); i > 0; i--)
      {
        auto& tc = iTrackingContexts[i - 1];

        if (tc.IsFrozen())
        {
          PurgeAndSaveTrackingContext(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
          //std::cout << "removed frozen tc\n";
          continue;
        }

        cv::Rect2d bb;

        bool fRet = tc.iTracker->update(m, bb);

        if (fRet)
        {
          if (IsRectInsideMat(bb, m))
          {
            tc.iTrail.push_back(bb);

            out.push_back(bb);

            if (!tc.iSkip)
            {
              tc.iSkip = iCounter->ProcessTrail(tc.iTrail, m);
            }
          }
          else
          {
            PurgeAndSaveTrackingContext(tc);
            iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
            //std::cout << "Tracker at " << (i - 1) << " is out of the bounds, size : " << iTrackingContexts.size() << "\n";
          }
        }
        else
        {
          PurgeAndSaveTrackingContext(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
          //std::cout << "Tracker at " << (i - 1) << " lost, size : " << iTrackingContexts.size() << "\n";
        }
      }

      return out;
    }

  protected:

};

using SPCTracker = std::shared_ptr<CTracker>;

#endif //TRACKER_HPP