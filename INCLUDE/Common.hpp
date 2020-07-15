#ifndef COMMON_HPP
#define COMMON_HPP 

bool IsRectInsideFrame(cv::Rect2d& r, cv::Mat& m)
{
  return ((static_cast<cv::Rect>(r) & cv::Rect(0, 0, m.cols, m.rows)) == static_cast<cv::Rect>(r));
}

#endif