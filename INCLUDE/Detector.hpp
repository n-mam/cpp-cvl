#ifndef DETECTOR_HPP
#define DETECTOR_HPP 

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class Detector
{
  public:

    Detector() {}

    Detector(const std::string& config, const std::string& weight)
    {
      iNetwork = cv::dnn::readNet(
        iConfigFile = "../MODELS/" + config, 
        iWeightFile = "../MODELS/" + weight
      );
      iNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      iNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    virtual ~Detector() {}
 
    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) = 0;

  protected:

    std::string iConfigFile;

    std::string iWeightFile;

    cv::dnn::Net iNetwork;
    
};

class FaceDetector : public Detector
{
  public:

    FaceDetector() :
     Detector("ResNetSSD_deploy.prototxt", "ResNetSSD_deploy.caffemodel") 
    {
    }

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      std::vector<cv::Rect2d> out;

      cv::Mat resized;

      cv::resize(frame, resized, cv::Size(300, 300));

      cv::Mat inputBlob = cv::dnn::blobFromImage(
        resized,
        1.0f,
        cv::Size(300, 300),
        cv::Scalar(104.0, 177.0, 123.0),
        false,
        false);

      iNetwork.setInput(inputBlob);

      cv::Mat detection = iNetwork.forward();

      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

      for (int i = 0; i < detectionMat.rows; i++)
      {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > 0.7)
        {
          int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
          int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
          int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
          int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

          out.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));

          std::cout << "Face detected at " << x1 << "," << y1 << "[" << x2 - x1 << "," << y2 - y1 << "]\n";
        }
      }

      return out;
    }
};

class ObjectDetector : public Detector
{
  public:

    ObjectDetector() :
     Detector("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel") 
    {
    }

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      std::vector<cv::Rect2d> out;

      cv::Mat resized;

      cv::resize(frame, resized, cv::Size(300, 300));

      cv::Mat inputBlob = cv::dnn::blobFromImage(
        resized,
        0.007843f,
        cv::Size(300, 300),
        cv::Scalar(127.5, 127.5, 127.5),
        false, 
        false);

      iNetwork.setInput(inputBlob);

      cv::Mat detection = iNetwork.forward();

      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

      for (int i = 0; i < detectionMat.rows; i++)
      {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > 0.7)
        {
          int idx, x1, y1, x2, y2;

          idx = static_cast<int>(detectionMat.at<float>(i, 1));

          if (iObjectClass[idx] == "person")
          {
            x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
            out.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
          }

          std::cout << "Object(" + iObjectClass[idx] + ") detected at " << x1 << "," << y1 << "[" << x2 - x1 << "," << y2 - y1 << "]\n";
        }
      }

      return out;
    }

  protected:

    std::string iObjectClass[21] = 
    {
      "background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"
    };
};

#endif