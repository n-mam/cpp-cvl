#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>

#include <tuple>
#include <vector>

bool IsRectInsideMat(const cv::Rect2d& r, const cv::Mat& m)
{
  return ((static_cast<cv::Rect>(r) & cv::Rect(0, 0, m.cols, m.rows)).area() == static_cast<cv::Rect>(r).area());
}

bool IsRectInsideRect(const cv::Rect2d& r1, const cv::Rect2d& r2)
{
  return (((static_cast<cv::Rect>(r1) & static_cast<cv::Rect>(r2))).area() == static_cast<cv::Rect>(r1).area());
}

bool DoesRectOverlapMat(const cv::Rect2d& r, const cv::Mat& m)
{
  return ((static_cast<cv::Rect>(r) & cv::Rect(0, 0, m.cols, m.rows)).area() > 0);
}

bool DoesRectOverlapRect(const cv::Rect2d& r1, const cv::Rect2d& r2)
{
  return (((static_cast<cv::Rect>(r1) & static_cast<cv::Rect>(r2))).area() > 0);
}

cv::Point GetRectCenter(const cv::Rect2d& r)
{
  return ((r.br() + r.tl()) * 0.5);
}

double Distance(cv::Point p1, cv::Point p2) //double x1, double y1, double x2, double y2)
{
  double x = p1.x - p2.x;
  double y = p1.y - p2.y;
  double dist = sqrt((x * x) + (y * y));
  return dist;
}

bool DoesIntersectReferenceLine(cv::Point start, cv::Point end, int refx, int refy)
{
  return false;
}

using Detection = std::tuple<cv::Rect2d, float, float, bool>;
using Detections = std::vector<Detection>;

struct TrackingContext
{
  int id;

  int iLostCount = 0;

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

    if (value > 0.5) 
    {
      _maleScore += value - 0.5f;
    }
    else
    {
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


using TOnCameraEventCbk = std::function<void (const std::string&, const std::string&, const std::string&, std::vector<uint8_t>&)>;

#endif

