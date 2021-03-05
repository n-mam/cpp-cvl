#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>
#include <functional>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <Counter.hpp>
#include <Geometry.hpp>

#include <CSubject.hpp>

class CTracker : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CTracker()
    {
      iCounter = std::make_shared<CCounter>();
    }

    ~CTracker()
    {
      ClearAllContexts();
    }

    void ClearAllContexts(void)
    {
      for (auto& tc : iTrackingContexts)
      {
        SaveAndPurgeTrackingContext(tc);
      }

      iTrackingContexts.clear();
      iCounter.reset(new CCounter());
    }

    size_t GetContextCount(void)
    {
      return iTrackingContexts.size();
    }

    void SetRefLine(int orientation, int delta)
    {
      iCounter->SetRefLine(orientation, delta);
    }

    virtual void RenderDisplacementAndPaths(cv::Mat& m, bool isTest = true)
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

    virtual void MatchDetectionWithTrackingContext(Detections& detections, cv::Mat& mat)
    {
      for (auto& t : iTrackingContexts)
      {
        int maxArea = 0;

        t.iMatch = nullptr;

        auto& last = t.iTrail.back();

        for (auto& d : detections)
        {
          auto& matched = std::get<3>(d);

          if (!matched)
          {
            auto& roi = std::get<0>(d);

            if (DoesRectOverlapRect(roi, last))
            {
              auto area = (roi & last).area();

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
          t.iThumbnails.emplace_back(mat(std::get<0>(*(t.iMatch))).clone());
        }
      }

      for (auto& t : iTrackingContexts)
      {
        if (t.iMatch)
        {
          cv::rectangle(mat, std::get<0>(*(t.iMatch)), 
               cv::Scalar(255, 0, 0 ), 1, 1);  // detection blue
        }
        else
        {
          t.iLostCount++;
        }
      }
    }

    virtual TrackingContext * AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& roi)
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

      cv::rectangle(m, roi, cv::Scalar(0, 0, 0 ), 2, 1);  // white

      iTrackingContexts.push_back(tc);

      return &(iTrackingContexts.back());
    }

    virtual std::vector<cv::Rect2d> UpdateTrackingContexts(cv::Mat& frame)
    {
      if (!iTrackingContexts.size())
      {
        return {};
      }

      std::vector<cv::Rect2d> out;

      for (size_t i = iTrackingContexts.size(); i > 0; i--)
      {
        auto& tc = iTrackingContexts[i - 1];

        if (tc.iLostCount > 10)
        {
          SaveAndPurgeTrackingContext(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
          std::cout << "removed frozen tc\n";
          continue;
        }

        cv::Rect2d bb;

        bool fRet = tc.iTracker->update(frame, bb);

        if (fRet)
        {
          if (IsRectInsideMat(bb, frame))
          {
            tc.iTrail.push_back(bb);

            out.push_back(bb);

            if (!tc.iSkip)
            {
              tc.iSkip = iCounter->ProcessTrail(tc.iTrail, frame);
            }
          }
          else
          {
            //std::cout << "Tracker " << tc.id << " out of the bound, trail size : " << tc.iTrail.size() << "\n";
            SaveAndPurgeTrackingContext(tc);
            iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
          }
        }
        else
        {
          //std::cout << "Tracker " << tc.id << " lost, trail size : " << tc.iTrail.size()<< "\n";
          SaveAndPurgeTrackingContext(tc);
          iTrackingContexts.erase(iTrackingContexts.begin() + (i - 1));
        }
      }

      return out;
    }

  protected:

    size_t iCount = 0;

    SPCCounter iCounter;

    std::vector<TrackingContext> iTrackingContexts;

    std::vector<TrackingContext> iPurgedContexts;

    virtual void SaveAndPurgeTrackingContext(TrackingContext& tc)
    {
      OnEvent(std::ref(tc));

      if (iPurgedContexts.size() > 20)
      {
        iPurgedContexts.clear();
      }

      iPurgedContexts.push_back(tc);
    }
};

using SPCTracker = std::shared_ptr<CTracker>;

#endif //TRACKER_HPP