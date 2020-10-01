#ifndef CVL_HPP
#define CVL_HPP

#ifdef OPENVINO
#include <CameraOV.hpp>
#else
#include <CameraCV.hpp>
#endif

namespace CVL 
{
 #ifdef OPENVINO
  auto make_camera(const std::string& source, const std::string& target)
  {
    return std::make_shared<CCamera>(source, target);
  }
 #else
  auto make_camera(const std::string& source, const std::string& target, const std::string& algo, const std::string& tracker)
  {
    return std::make_shared<CCamera>(source, target, algo, tracker);
  }
 #endif

}

#endif