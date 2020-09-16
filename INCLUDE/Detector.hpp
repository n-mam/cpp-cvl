#ifndef DETECTOR_HPP
#define DETECTOR_HPP 

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <CSubject.hpp>

class CDetector : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CDetector() {}

    CDetector(const std::string& config, const std::string& weight)
    {
      char *isTest = getenv("cpp-cvl-test");

      if (isTest)
      {
        iConfigFile = "../../cpp-cvl/MODELS/" + config;
        iWeightFile = "../../cpp-cvl/MODELS/" + weight;
      }
      else
      {
        iConfigFile = "./MODELS/" + config;
        iWeightFile = "./MODELS/" + weight;
      }

      try
      {
        iNetwork = cv::dnn::readNet(iConfigFile, iWeightFile);
      }
      catch(const std::exception& e)
      {
        std::cerr << e.what() << " : readNet failed\n";
      }

      // try
      // {
      //   iNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
      //   iNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      // }
      // catch(const std::exception& e)
      // {
      //   std::cerr << e.what() << "\n";
      //   iNetwork.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
      //   iNetwork.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
      // }      
    }

    virtual ~CDetector() {}
 
    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) = 0;

  protected:

    std::string iTarget;

    std::string iConfigFile;

    std::string iWeightFile;

    cv::dnn::Net iNetwork;
};

using SPCDetector = std::shared_ptr<CDetector>;

class AgeDetector : public CDetector
{
  public:

    AgeDetector() :
     CDetector("age_net.caffemodel", "deploy_age.prototxt") 
    {
    }

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      auto blob = cv::dnn::blobFromImage(
        frame, 1, cv::Size(227, 227), 
        cv::Scalar(78.4263377603, 87.7689143744, 114.895847746), 
        false);
      iNetwork.setInput(blob);
      std::vector<float> agePreds = iNetwork.forward();
      int max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
      std::string age = ageList[max_indice_age];
      std::cout << "age : " << age << "\n";
      return {};
    }
  
  protected:

    std::vector<std::string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
};

class GenderDetector : public CDetector
{
  public:

    GenderDetector() :
     CDetector("gender_net.caffemodel", "deploy_gender.prototxt") 
    {
    }

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      auto blob = cv::dnn::blobFromImage(
        frame, 1, cv::Size(227, 227), 
        cv::Scalar(78.4263377603, 87.7689143744, 114.895847746), 
        false);
      iNetwork.setInput(blob);
      std::vector<float> genderPreds = iNetwork.forward();
      int max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
      std::string gender = genderList[max_index_gender];
      std::cout << "gender : " << gender << "\n";
      return {};
    }

  protected:

    std::vector<std::string> genderList = {"Male", "Female"};
};

class FaceDetector : public CDetector
{
  public:

    FaceDetector() :
     CDetector("ResNetSSD_deploy.prototxt", "ResNetSSD_deploy.caffemodel") 
    {
      iAgeDetector = std::make_shared<AgeDetector>();
      iGenderDetector = std::make_shared<GenderDetector>();
    }

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      std::vector<cv::Rect2d> out;

      cv::Mat inputBlob = cv::dnn::blobFromImage(
        frame,
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

          auto rect = cv::Rect2d(x1, y1, x2-x1, y2-y1);

          if (IsRectInsideMat(rect, frame))
          {
            cv::Mat roi = cv::Mat(frame, rect);

            if (iAgeDetector)
            {
              iAgeDetector->Detect(roi);
            }

            if (iGenderDetector)
            {
              iGenderDetector->Detect(roi);
            }
          }
        }
      }

      return out;
    }

  protected:

    SPCDetector iAgeDetector = nullptr;
    SPCDetector iGenderDetector = nullptr;
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

    BackgroundSubtractor() : CDetector()
    {
      SetProperty("bbarea", "100");
      SetProperty("exhzbb", "false");
    }

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

      auto areaThreshold = GetPropertyAsInt("bbarea");
      auto excludeHBB = GetPropertyAsInt("exhzbb");

      for (size_t i = 0; i < contours.size(); i++) 
      {
        if (cv::contourArea(contours[i]) < areaThreshold)
        {
          continue;
        }

        auto bb = cv::boundingRect(contours[i]);

        if (excludeHBB && (bb.width > bb.height)) 
        {
          continue;
        }

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

    virtual void SetProperty(const std::string& key, const std::string& value) override
    {
      if (key == "MarkBaseFrame")
      {
        iFirstFrame.release();
      }
      CSubject<uint8_t, uint8_t>::SetProperty(key, value);
    }

  protected:

    cv::Mat iFirstFrame;
};

#endif