#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <Common.hpp>

struct TrackinContext
{
  cv::Ptr<cv::Tracker> iTracker;  // cv tracker     
  std::vector<cv::Rect2d> iBBTrail;  // last bb
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

    void AddNewTrackingContext(cv::Mat& m, cv::Rect2d& r)
    {
      TrackinContext tc;

      tc.iTracker = cv::TrackerCSRT::create();

      tc.iBBTrail.push_back(r);

      tc.iTracker->init(m, r);

      iTrackingContexts.push_back(tc);
    }

    void ClearAllContexts(void)
    {
      iTrackingContexts.clear();
    }

    void Update(cv::Mat& frame)
    {
      if (!iTrackingContexts.size()) return;
      
      for (size_t i = (iTrackingContexts.size() - 1); i >= 0; i--)
      {
        auto& tc = iTrackingContexts[i];

        cv::Rect2d bb;

        bool fRet = tc.iTracker->update(frame, bb);

        if (fRet)
        {
          if (IsRectInsideMat(bb, frame))
          {
            tc.iBBTrail.push_back(bb);

            for(auto& b : tc.iBBTrail)
            {
              cv::circle(frame, b.br(), 1, cv::Scalar(255,0,0));
            }

            cv::rectangle(frame, bb, cv::Scalar(255, 0, 0 ), -1, 1);
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

    std::vector<TrackinContext> iTrackingContexts;
};


#endif //TRACKER_HPP