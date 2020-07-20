#ifndef TRACKER_HPP
#define TRACKER_HPP 

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>

#include <Common.hpp>

struct TrackinContext
{
  cv::Ptr<cv::Tracker> iTracker;  // cv tracker
  dlib::correlation_tracker tracker; //dlib tracker
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

    void ClearAllContexts(void)
    {
      iTrackingContexts.clear();
    }

    virtual void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) {}

    virtual void Update(const cv::Mat& frame) {}

  protected:

    std::vector<TrackinContext> iTrackingContexts;
};

class DlibTracker : public TrackingManager
{
  public:

    void AddNewTrackingContext(const cv::Mat& m, cv::Rect2d& r) override
    {
      TrackinContext tc;

      tc.iBBTrail.push_back(r);
//dlib::bgr_pixel
      dlib::array2d<dlib::bgr_pixel> img;
      dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(m));

      tc.tracker.start_track(img, dlib::centered_rect(dlib::point(r.x, r.y), r.width, r.height));

      iTrackingContexts.push_back(tc);
    }

    void Update(const cv::Mat& m) override
    {
      if (!iTrackingContexts.size()) return;

      for (int i = (iTrackingContexts.size() - 1); i >= 0; i--)
      {
        auto& tc = iTrackingContexts[i];

        dlib::array2d<dlib::bgr_pixel> img;
        dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(m));

        tc.tracker.update(img);
      }
    }

  protected:

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

    void Update(const cv::Mat& frame) override
    {
      if (!iTrackingContexts.size()) return;

      for (int i = (iTrackingContexts.size() - 1); i >= 0; i--)
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

};


#endif //TRACKER_HPP