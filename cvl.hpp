#ifndef CVL_HPP
#define CVL_HPP

#include <Camera.hpp>

namespace CVL 
{

auto make_camera(const std::string& source, const std::string& target, const std::string& algo, const std::string& tracker)
{
  return std::make_shared<CCamera>(source, target, algo, tracker);
}

}

#endif