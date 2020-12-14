#ifndef CVL_HPP
#define CVL_HPP

#include <CameraOV.hpp>
#include <CameraCV.hpp>

namespace CVL 
{
  auto make_camera(const std::string& source, const std::string& target, const std::string& algo, const std::string& tracker)
  {
    return std::make_shared<CCamera>(source, target, algo);
  }

  auto make_camera_ov(const std::string& source, const std::string& target)
  {
    return std::make_shared<COVCamera>(source, target);
  }
}

#endif