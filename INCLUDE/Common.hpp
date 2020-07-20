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

#endif

