#include <functional>
#include <iostream>
#include <vector>
#include <string>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

int main(void)
{
  InferenceEngine::Core core;
  
  auto network = core.ReadNetwork("Model.xml");
  
  
  
  
  
  return 0;
}
