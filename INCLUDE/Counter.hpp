#ifndef COUNTER_HPP
#define COUNTER_HPP

#include <tuple>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

#include <Geometry.hpp>

class CCounter
{
  public:

    CCounter()
    {

    }

    ~CCounter()
    {
      up = down = left = right = 0;
      iTrackingContextTrail.clear();
    }

    void Add(const std::vector<cv::Rect2d>& trail)
    {
      iTrackingContextTrail.push_back(trail);
    }

    bool ProcessTrail(const std::vector<cv::Rect2d>& trail, cv::Mat m)
    {
      auto start = GetRectCenter(trail.front());
      auto end = GetRectCenter(trail.back());

      auto [u, d, l, r] = IsPathIntersectingRefLine(start, end, m);

      if (u || d || l || r)
      {
        up += u;
        down += d;
        left += l;
        right += r;
        return true;
      }

      return false;
    }

    cv::Point GetRefStartPoint(cv::Mat& m)
    {
      if (iRefOrientation) // H
      {
        return cv::Point(0, m.rows/2 + iRefDelta);
      }
      else // V
      {
        return cv::Point(m.cols/2 + iRefDelta, 0);
      }
    }

    cv::Point GetRefEndPoint(cv::Mat& m)
    {
      if (iRefOrientation) // H
      {
        return cv::Point(m.cols, m.rows/2 + iRefDelta);
      }
      else // V
      {
        return cv::Point(m.cols/2 + iRefDelta, m.rows);
      }
    }

    void SetRefLine(int orientation, int delta)
    {
      iRefOrientation = orientation;
      iRefDelta += delta;
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

    void DisplayCounts(const cv::Mat& m)
    {
      cv::putText(m, "u : " + std::to_string(up), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
      cv::putText(m, "d : " + std::to_string(down), cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
      cv::putText(m, "l : " + std::to_string(left), cv::Point(5, 70), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
      cv::putText(m, "r : " + std::to_string(right), cv::Point(5, 90), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }

  protected:

    std::vector <
      std::vector<cv::Rect2d>
    > iTrackingContextTrail;

    int up = 0;
    int down = 0;
    int left = 0;
    int right = 0;

    char iRefOrientation = 1; // horizontal

    int iRefDelta = 0; // ref line offset from baseline
};

using SPCCounter = std::shared_ptr<CCounter>;

#endif