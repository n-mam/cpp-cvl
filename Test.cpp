#include <iostream>

#include <cvl.hpp>

int main(int argc, char *argv[])
{
  auto camera = CVL::make_camera(argv[1], "person", "CSRT");

  camera->SetName("CV");

  camera->Start(
    [](const std::string& e, const std::string& data) {
      std::cout << "Camera event callback : " << e << " data : " << data << "\n";
    });

  getchar();

  camera->Stop();
}
