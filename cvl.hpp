#ifndef CVL_HPP
#define CVL_HPP

#include <Camera.hpp>

namespace CVL 
{

auto make_camera(const std::string& rtsp)
{
  return std::make_shared<CCamera>(rtsp);
}



}

#endif