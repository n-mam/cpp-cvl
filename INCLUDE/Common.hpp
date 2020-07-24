#ifndef COMMON_HPP
#define COMMON_HPP 

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

#endif

