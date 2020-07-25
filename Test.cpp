#include <iostream>

#include <cvl.hpp>

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  auto camera = CVL::make_camera(argv[1]);

  camera->Start();

  getchar();
}
