#ifndef CVL_HPP
#define CVL_HPP

#ifdef OPENVINO
#include <CameraOV.hpp>
#else
#include <CameraCV.hpp>
#endif


namespace CVL 
{

  auto make_camera(const std::string& source, const std::string& target, const std::string& algo, const std::string& tracker)
  {
    #ifdef OPENVINO
      return std::make_shared<CCamera>(source, target);
    #else
      return std::make_shared<CCamera>(source, target, algo, tracker);
    #endif
  }

}

#endif