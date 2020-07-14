#define CAFFE
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

int nPeople = 0;
std::vector<cv::Ptr<cv::Tracker>> Trackers;
const std::string caffeConfigFile = "../data/deploy.prototxt";
const std::string caffeWeightFile = "../data/res10_300x300_ssd_iter_140000.caffemodel";

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

    if (confidence > 0.7)
    {
      int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
      int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
      int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
      int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

      cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1, 4);
      cv::putText(frame, std::to_string(confidence), cv::Point(x1, y1-10), cv::FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1);

      cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

      tracker->init(frame, cv::Rect2d(cv::Point(x1, y1), cv::Point(x2, y2)));

      Trackers.push_back(tracker);

      nPeople++;

      std::cout << "Detection " << x1 << "," << y1 << "[" << x2 - x1 << "," << y2 - y1 << "] added. " 
                << "Total trackers : " << Trackers.size() << "\n";
    }
  }
}

int main(int argc, char *argv[])
{
  #if WIN32
    _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");
  #else
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");
  #endif

  auto net = cv::dnn::readNet(caffeWeightFile, caffeConfigFile);

  cv::VideoCapture cap("c:/code/P1033658.mp4"); //, cv::CAP_FFMPEG); 0, cv::CAP_DSHOW ); //P1033658

  if(!cap.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  cv::namedWindow("Display Window");

  cv::Mat frame;
  uint64_t count = 0;
  uint64_t nTotal = cap.get(cv::CAP_PROP_FRAME_COUNT);

  for (;;)
  {
    if(!cap.read(frame))
	  {
      if(count == nTotal)
      {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        Trackers.clear();
        count = 0;
        nPeople = 0;
        continue;
      }
      else
      {
        break;
      }
    }

    count++;

    auto w = frame.cols;
    auto h = frame.rows;

    auto scale = (float) 600 / frame.cols;
    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);
    // check if the detector reqires grayscale
    /*
     * first update all active trackers
     */
    for (int i = (Trackers.size() - 1); i >=0; i--)
    {
      cv::Rect2d bb;

      auto t = Trackers[i];

      bool fRet = t->update(frame, bb);

      if (fRet)
      {
        cv::rectangle(frame, bb, cv::Scalar(255, 0, 0 ), -1, 1);
      }
      else
      {
        Trackers.erase(Trackers.begin() + i);

        std::cout << "Tracker at " << i << " lost, size : " << Trackers.size() << "\n";
      }
    }

    /*
     * Now detect all faces using the updated (masked) 
     * frame. That way, only new detections would happen
     */
    if (count % 8 == 0)
    {
      DetectFacesDNN(net, frame);
    }

    cv::putText(frame, std::to_string(nPeople), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("Display Window", frame);

    if(cv::waitKey(1) >= 0) 
	  {
	    break;
	  }
  }   
}