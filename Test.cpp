#include <iostream>

#include <cvl.hpp>

int main(int argc, char *argv[])
{
  std::string source = argv[1];

  #ifdef OPENVINO

   auto camera = CVL::make_camera(source, "face");
   
   camera->SetName("CV");

   camera->Start(
    [](const std::string& e, const std::string& data, std::vector<uint8_t>& frame) {
      //std::cout << "Camera event callback : " << e << " data : " << data << "\n";
    }
   );

   getchar();

   camera->Stop();

  #else

   putenv("cpp-cvl-test=true");

   auto camera = CVL::make_camera(source, "person", "gmg", "CSRT");
   //auto camera = CVL::make_camera(source, "face", "gmg", "CSRT");

   camera->SetName("CV");

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