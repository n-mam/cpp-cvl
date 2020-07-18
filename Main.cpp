#define CAFFE
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/dnn.hpp>

#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>

uint64_t people = 0;

const std::string caffeConfigFile = "../data/deploy.prototxt";
const std::string caffeWeightFile = "../data/res10_300x300_ssd_iter_140000.caffemodel";

TrackingManager tm;

void DetectFacesDNN(cv::dnn::Net& net, cv::Mat& frame)
{
  int frameHeight = frame.rows;
  int frameWidth = frame.cols;

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(300, 300));

  cv::Mat inputBlob = cv::dnn::blobFromImage(
    resized,
    1.0f,
    cv::Size(300, 300), //model is 300x300
    cv::Scalar(104.0, 177.0, 123.0),
    false, //caffe uses RBG now ?
    false);

  net.setInput(inputBlob, "data");
  
  cv::Mat detection = net.forward("detection_out");

  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

  for (int i = 0; i < detectionMat.rows; i++)
  {
    float confidence = detectionMat.at<float>(i, 2);

    if (confidence > 0.2)
    {
      int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
      int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
      int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
      int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

      //tm.AddNewTrackingContext(frame, cv::Rect2d(cv::Point(x1, y1), cv::Point(x2, y2)));
      cv::rectangle(frame, cv::Rect2d(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(255, 0, 0 ), 1, 1);
      people++;

      std::cout << "Detection " << x1 << "," << y1 << "[" << x2 - x1 << "," << y2 - y1 << "] added. " 
                << "Total tracking contexts : " << tm.GetContextCount() << "\n";
    }
  }
}

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  auto net = cv::dnn::readNet(caffeWeightFile, caffeConfigFile);

  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

  Source s("f:/my.mp4");

  if(!s.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  cv::namedWindow("People Counting");

  cv::Mat frame;

  for (;;)
  {
    if(!s.Read(frame))
	  {
      if(s.HasEnded())
      {
        s.Rewind();
        tm.ClearAllContexts();
        people = 0;
        continue;
      }
      else
      {
        break;
      }
    }

    auto scale = (float) 600 / frame.cols;

    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);

    if (frame.channels() == 4)
    {
      cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    /*
     * update all active trackers first
     */
    tm.Update(frame);
    /*
     * Now detect all faces using the updated (masked) 
     * frame. That way, only new detections would happen
     */
    if (s.GetCurrentOffset() % 4 == 0)
    {
      DetectFacesDNN(net, frame);
    }

    cv::putText(frame, std::to_string(people), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("People Counting", frame);

    if (!s.HandleUserInput()) break;
  }   
}
