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

  std::vector<cv::Mat> iThumbnails;

  std::vector<float> iAge;

  float _maleScore = 0;
  float _femaleScore = 0;

  std::vector<std::string> iGender;

  bool iSkip = false;

  uint32_t lost = 0;

  bool IsFrozen(void)
  {/*
    * valid only for FOV where the subject is 
    * moving eithe top-down or left to right
    */
    if (iTrail.size() >= 8)
    {
      auto& last = iTrail[iTrail.size() - 1];
      auto& prev = iTrail[iTrail.size() - 8];
      auto d = Distance(GetRectCenter(last), GetRectCenter(prev));
      if (d <= 2) return true;
    }
    return false;
  }

  void updateAge(float value)
  {
    if (value < 0)
      return;
    float _age = (iAge.size() == 0) ? value : 0.95f * iAge.back() + 0.05f * value;
    iAge.push_back(_age);
  }

  void updateGender(float value) 
  {
    if (value < 0)
      return;
    if (value > 0.5) {
      _maleScore += value - 0.5f;
    } else {
      _femaleScore += 0.5f - value;
    }

    iGender.push_back(isMale() ? "M" : "F");
  }

  int getAge() 
  {
    return static_cast<int>(std::floor(iAge.back() + 0.5f));
  }

  bool isMale() 
  {
    return _maleScore > _femaleScore;
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

        if (tc.iAge.size()) {
          auto ag = (tc.isMale() ? std::string("M") : std::string("F")) + ":" + std::to_string(tc.getAge());
          cv::putText(m, ag,
               cv::Point((int)bb.x, (int)(bb.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 
               0.4, cv::Scalar(255, 255, 255), 1);
        }

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

    TrackingContext * MatchDetectionWithTrackingContext(cv::Rect2d& roi, cv::Mat& mat)
    {
      for (auto& tc : iTrackingContexts)
      {
        if (DoesRectOverlapRect(roi, tc.iTrail.back()))
        {
          if (IsRectInsideMat(roi, mat))
          {
            tc.iThumbnails.push_back(mat(roi).clone());
          }

          cv::rectangle(mat, roi, cv::Scalar(255, 0, 0 ), 1, 1);  // detection blue

          return &tc;
        }
      }
      return nullptr;
    }

    virtual TrackingContext * AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) { return nullptr; }

    virtual std::vector<cv::Rect2d> UpdateTrackingContexts(cv::Mat& frame) { return {}; }

    virtual void SetEventCallback(TOnCameraEventCbk cbk)
    {
      iOnCameraEventCbk = cbk;
    }

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
        std::string path;

        for (auto& r : tc.iTrail)
        {
          auto p = GetRectCenter(r);

          if (path.size())
          {
            path += ", ";
          }

          path += std::to_string(p.x) + " " + std::to_string(p.y);
        }

        if (tc.iAge.size() != tc.iGender.size()) 
        {
          throw std::exception("age-gender size mismatch");
        }

        std::string demography;

        for (int i = 0; i < tc.iAge.size(); i++)
        {
          if (demography.size())
          {
            demography += ", ";
          }

          demography += std::to_string((int)tc.iAge[i]) + " " + tc.iGender[i];         
        }

        if (demography.size())
        {
          demography += std::string(", ") + std::to_string(tc.getAge()) + std::string(" ") + (tc.isMale() ? std::string("M") : std::string("F"));
        }

        std::vector<uchar> thumb;

        if (tc.iThumbnails.size())
        {
          cv::imencode(".jpg", tc.iThumbnails[tc.iThumbnails.size()/2], thumb);
        }

        iOnCameraEventCbk("trail", path, demography, thumb /*std::vector<uint8_t>()*/);
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

    TrackingContext * AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& roi) override
    {
      if (roi.x + roi.width > m.cols) return nullptr; //roi.width -= m.cols - roi.x;
      if (roi.y + roi.height > m.rows) return nullptr; //roi.height -= m.rows - roi.y;

      TrackingContext tc;

      cv::TrackerCSRT::Params params;
      params.psr_threshold = 0.04f;

      tc.iTracker = cv::TrackerCSRT::create(params);

      tc.iTrail.push_back(roi);

      tc.iTracker->init(m, roi);

      iTrackingContexts.push_back(tc);

      return &(iTrackingContexts.back());
    }

    std::vector<cv::Rect2d> UpdateTrackingContexts(cv::Mat& m) override
    {
      if (!iTrackingContexts.size())
      {
        return {};
      }

      std::vector<cv::Rect2d> out;

      for (size_t i = iTrackingContexts.size(); i > 0; i--)
      {
        auto& tc = iTrackingContexts[i - 1];

        // if (tc.IsFrozen())
        // {
        //   PurgeAndSaveTrackingContext(tc);
        //   iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
        //   //std::cout << "removed frozen tc\n";
        //   continue;
        // }

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
            std::cout << "Tracker at " << (i - 1) << " out of the bound, size : " << tc.iTrail.size() << "\n";
            PurgeAndSaveTrackingContext(tc);
            iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
          }
        }
        else
        {
          std::cout << "Tracker at " << (i - 1) << " lost, size : " << iTrackingContexts.size() << "\n";
          PurgeAndSaveTrackingContext(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
        }
      }

      return out;
    }

  protected:

};

using SPCTracker = std::shared_ptr<CTracker>;

#endif //TRACKER_HPP