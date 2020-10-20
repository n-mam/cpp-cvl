#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <Counter.hpp>
#include <Geometry.hpp>

using Detection = std::tuple<cv::Rect2d, float, float, bool>;
using Detections = std::vector<Detection>;

struct TrackingContext
{
  int id;

  Detection *iMatch;

  std::vector<float> iAge;

  std::vector<std::string> iGender;
  float _maleScore = 0;
  float _femaleScore = 0;

  cv::Ptr<cv::Tracker> iTracker;   // cv tracker

  std::vector<cv::Rect2d> iTrail;  // track bb trail

  std::vector<cv::Mat> iThumbnails;

  bool iSkip = false;

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

    void RenderDisplacementAndPaths(cv::Mat& m, bool isTest = true)
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
        for (size_t i = 1; i < tc.iTrail.size() - 1; ++i)
        {
          auto&& f = GetRectCenter(tc.iTrail[i]);
          auto&& b = GetRectCenter(tc.iTrail[i-1]);
          cv::line(m, f, b, cv::Scalar(0, 255, 0), 1);
        }

        auto& bb = tc.iTrail.back();

        if (tc.iAge.size()) {
          auto&& ag = (tc.isMale() ? std::string("M") : std::string("F")) + ":" + std::to_string(tc.getAge());
          cv::putText(m, ag, cv::Point((int)bb.x, (int)(bb.y - 5)), 
             cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }

        cv::rectangle(m, bb, cv::Scalar(0, 0, 255), 1, 1); //tracking red
      }

      for (auto& pc : iPurgedContexts)
      {
        auto&& first = GetRectCenter(pc.iTrail.front());
        auto&& last = GetRectCenter(pc.iTrail.back());
        cv::line(m, first, last, cv::Scalar(0, 255, 255), 1);
      }

      if (isTest) iCounter->DisplayRefLineAndCounts(m);
    }

    void MatchDetectionWithTrackingContext(Detections& detections, cv::Mat& mat)
    {
      for (auto& t : iTrackingContexts)
      {
        int maxArea = 0;
        t.iMatch = nullptr;

        for (auto& d : detections)
        {
          auto& roi = std::get<0>(d);
          auto& matched = std::get<3>(d);

          if (!matched)
          {
            if (DoesRectOverlapRect(roi, t.iTrail.back()))
            {
              auto area = (roi & t.iTrail.back()).area();

              if (area > maxArea)
              {
                maxArea = area;
                t.iMatch = &d;
              }
            }
          }
        }

        if (t.iMatch)
        {
          std::get<3>(*(t.iMatch)) = true;
          t.iThumbnails.push_back(mat(std::get<0>(*(t.iMatch))).clone());
        }
      }

      for (auto& t : iTrackingContexts)
      {
        if (t.iMatch)
        {
          cv::rectangle(mat, std::get<0>(*(t.iMatch)), 
               cv::Scalar(255, 0, 0 ), 1, 1);  // detection blue
        }
      }
    }

    TrackingContext * MatchDetectionWithTrackingContextOld(cv::Rect2d& roi, cv::Mat& mat)
    {
      int maxArea = 0, maxIndex = -1;

      for (int i = 0; i < iTrackingContexts.size(); ++i)
      {
        auto& tc = iTrackingContexts[i];

        if (DoesRectOverlapRect(roi, tc.iTrail.back()))
        {
          auto intersection = roi & tc.iTrail.back();

          auto area = intersection.width * intersection.width;

          if (area > maxArea)
          {
            maxArea = area;
            maxIndex = i;
          }
        }
      }

      if (maxIndex >= 0)
      {
        auto& tc = iTrackingContexts[maxIndex];

        if (IsRectInsideMat(roi, mat))
        {
          tc.iThumbnails.push_back(mat(roi).clone());
        }

        cv::rectangle(mat, roi, cv::Scalar(255, 0, 0 ), 1, 1);  // detection blue

        return &tc;
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

    size_t iCount = 0;

    std::string iType;

    SPCCounter iCounter;

    TOnCameraEventCbk iOnCameraEventCbk = nullptr;

    std::vector<TrackingContext> iTrackingContexts;

    std::vector<TrackingContext> iPurgedContexts;

    virtual void PurgeAndSaveTrackingContext(TrackingContext& tc)
    {
      if (iOnCameraEventCbk && tc.iThumbnails.size())
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

        for (int i = 0; i < tc.iAge.size(); ++i)
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
        cv::imencode(".jpg", tc.iThumbnails[tc.iThumbnails.size()/2], thumb);
        //cv::imwrite("some.jpg", tc.iThumbnails[tc.iThumbnails.size()/2]);

        iOnCameraEventCbk("trail", path, demography, thumb);
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
      for (auto& tc : iTrackingContexts)
      {
        if ((roi & tc.iTrail.back()).area()) return nullptr;
      }

      TrackingContext tc;
      tc.id = iCount++;

      cv::TrackerCSRT::Params params;
      params.psr_threshold = 0.04f; //0.035f; 
      //param.template_size = 150;
      //param.admm_iterations = 3;

      tc.iTracker = cv::TrackerCSRT::create(params);

      tc.iTrail.push_back(roi);

      tc.iTracker->init(m, roi);

      cv::rectangle(m, roi, cv::Scalar(0, 0, 0 ), 2, 1);  // 

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
            //std::cout << "Tracker " << tc.id << " out of the bound, trail size : " << tc.iTrail.size() << "\n";
            PurgeAndSaveTrackingContext(tc);
            iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
          }
        }
        else
        {
          //std::cout << "Tracker " << tc.id << " lost, trail size : " << tc.iTrail.size()<< "\n";
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