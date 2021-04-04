#include <iostream>

#include <cvl.hpp>

int main(int argc, char *argv[])
{
  std::string target = argv[1];
  std::string source = argv[2];

  #ifdef OPENVINO

   auto camera = CVL::make_camera(source, target);
   
   camera->SetProperty("name", "CV");

   camera->Start(
    [](const std::string& e, const std::string& data, std::vector<uint8_t>& frame) {
      //std::cout << "Camera event callback : " << e << " data : " << data << "\n";
    }
   );

   getchar();

   camera->Stop();

  #else

   putenv("cpp-cvl-test=true");

   auto camera = CVL::make_camera(source, target, "gmg", "CSRT");

   camera->SetProperty("name", "CV");
   camera->SetProperty("skipcount", "0");
   camera->SetProperty("bbarea", "1000");

   camera->Start(
    [](const std::string& e, const std::string& path, const std::string& demography, std::vector<uint8_t>& frame) {
      std::cout << "\nCamera event callback [" << e << "]" << " Path : " << path << " Demography : " << demography << "\n";
    }
   );

   getchar();

   camera->Stop();

  #endif

}