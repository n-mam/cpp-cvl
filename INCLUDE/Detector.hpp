#ifndef DETECTOR_HPP
#define DETECTOR_HPP 

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class CDetector
{
  public:

    CDetector() {}

    CDetector(const std::string& config, const std::string& weight)
    {
      iNetwork = cv::dnn::readNet(
        iConfigFile = "../../cpp-cvl/MODELS/" + config, 
        iWeightFile = "../../cpp-cvl/MODELS/" + weight
      );
      iNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      iNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    virtual ~CDetector() {}
 
    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) = 0;

  protected:

    std::string iTarget;

    std::string iConfigFile;

    std::string iWeightFile;

    cv::dnn::Net iNetwork;
    
};

class FaceDetector : public CDetector
{
  public:

    FaceDetector() :
     CDetector("ResNetSSD_deploy.prototxt", "ResNetSSD_deploy.caffemodel") 
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

class ObjectDetector : public CDetector
{
  public:

    ObjectDetector(const std::string& target) :
     CDetector("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel") 
    {
      iTarget = target;
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

          if (iObjectClass[idx] == iTarget)
          {
            x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
            out.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
            std::cout << "Object(" + iObjectClass[idx] + ") detected at " << x1 << "," << y1 << "[" << x2 - x1 << "," << y2 - y1 << "]\n";
          }      
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

class BackgroundSubtractor : public CDetector
{
  public:

    BackgroundSubtractor() : CDetector() {}

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      cv::Mat gray, blur, delta, thresh, dilate;

      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      cv::GaussianBlur(gray, blur, cv::Size(21,21), 0);

      if (iFirstFrame.empty())
      {
        iFirstFrame = blur;
      }

	    cv::absdiff(iFirstFrame, blur, delta);
	    cv::threshold(delta, thresh, 25, 255, cv::THRESH_BINARY);
      cv::dilate(thresh, dilate, cv::Mat(), cv::Point(-1, -1), 2);

      std::vector<
        std::vector<cv::Point>
      > contours;

      std::vector<cv::Rect2d> detections;

      cv::findContours(dilate, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

      for (size_t i = 0; i < contours.size(); i++) 
      {
        if (cv::contourArea(contours[i]) < 1000) continue;

        auto bb = cv::boundingRect(contours[i]);
        if (bb.width > bb.height) continue;  

        bool skip = false;

        for (size_t j = 0; j < contours.size(); j++)
        {
          if (DoesRectOverlapRect(bb, cv::boundingRect(contours[j])))
          {
            if (i != j)
            {
              skip = true;
              break;
            }
          }
        }

        if (!skip)
        {
          detections.push_back(bb);
        }
      }
      return detections;
    }
  
  protected:

    cv::Mat iFirstFrame;

};

using SPCDetector = std::shared_ptr<CDetector>;

#endif