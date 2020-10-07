#ifndef DETECTOR_HPP
#define DETECTOR_HPP 

#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <opencv2/bgsegm.hpp>

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

    	if (cv::cuda::getCudaEnabledDeviceCount())
	    {
		    iNetwork.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
		    iNetwork.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
		    std::cout << "CUDA backend and target enabled for inference." << std::endl;
	    }
	    else
	    {
		    iNetwork.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_INFERENCE_ENGINE);
		    iNetwork.setPreferableTarget(cv::dnn::Target::DNN_TARGET_OPENCL_FP16);
		    std::cout << "IE backend and cpu target enabled for inference." << std::endl;
	    }
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

    AgeDetector() : CDetector("age_net.caffemodel", "deploy_age.prototxt") 
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
      auto max_indice_age = std::distance(agePreds.begin(), max_element(agePreds.begin(), agePreds.end()));
      iAge.push_back(ageList[max_indice_age]);
      std::cout << "age : " << iAge.back() << "\n";
      return {};
    }

  public:

    std::vector<std::string> iAge;

  protected:

    std::vector<std::string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
};

class GenderDetector : public CDetector
{
  public:

    GenderDetector() : CDetector("gender_net.caffemodel", "deploy_gender.prototxt") 
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
      auto max_index_gender = std::distance(genderPreds.begin(), max_element(genderPreds.begin(), genderPreds.end()));
      iGender.push_back(genderList[max_index_gender]);
      std::cout << "gender : " << iGender.back() << "\n";
      return {};
    }

  public:

    std::vector<std::string> iGender;

  protected:

    std::vector<std::string> genderList = {"Male", "Female"};
};

using SPAgeDetector = std::shared_ptr<AgeDetector>;
using SPGenderDetector = std::shared_ptr<GenderDetector>;

class FaceDetector : public CDetector
{
  public:

    FaceDetector() : CDetector("ResNetSSD_deploy.prototxt", "ResNetSSD_deploy.caffemodel") 
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

        if (confidence > 0.6)
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

            cv::putText(frame, (iGenderDetector->iGender).back() + (iAgeDetector->iAge).back(),
               cv::Point((int)rect.x, (int)(rect.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 
               0.5, cv::Scalar(0, 0, 255), 1);
          }
        }
      }

      return out;
    }

  protected:

    SPAgeDetector iAgeDetector = nullptr;
    SPGenderDetector iGenderDetector = nullptr;
};

class ObjectDetector : public CDetector
{
  public:

    ObjectDetector(const std::string& target) :
     //CDetector("person-detection-retail-0013/FP16/person-detection-retail-0013.bin", "person-detection-retail-0013/FP16/person-detection-retail-0013.xml") 
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

        if (confidence > 0.6)
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
            //std::cout << "Object(" + iObjectClass[idx] + ") detected at " << x1 << "," << y1 << "[" << x2 - x1 << "," << y2 - y1 << "]\n";
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

    BackgroundSubtractor(const std::string& algo) : CDetector()
    {
      SetProperty("bbarea", "10");
      SetProperty("exhzbb", "1");

      if (algo == "mog") {
         pBackgroundSubtractor = cv::bgsegm::createBackgroundSubtractorMOG();
      } else if (algo == "cnt") {
         pBackgroundSubtractor = cv::bgsegm::createBackgroundSubtractorCNT();
      } else if (algo == "gmg") {
         pBackgroundSubtractor = cv::bgsegm::createBackgroundSubtractorGMG();
      } else if (algo == "gsoc") {
         pBackgroundSubtractor = cv::bgsegm::createBackgroundSubtractorGSOC();
      } else if (algo == "lsbp") {
         pBackgroundSubtractor = cv::bgsegm::createBackgroundSubtractorLSBP();
      } else {
         pBackgroundSubtractor = cv::bgsegm::createBackgroundSubtractorGMG();
      }
    }

    virtual std::vector<cv::Rect2d> Detect(cv::Mat& frame) override
    {
      cv::Mat fgMask;

      std::vector<
        std::vector<cv::Point>
      > contours;

      std::vector<cv::Rect2d> detections;

      pBackgroundSubtractor->apply(frame, fgMask, 0.8);

      cv::findContours(fgMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

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
          cv::putText(frame, std::to_string((int)(bb.width * bb.height)),
             cv::Point((int)bb.x, (int)(bb.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 
             0.5, cv::Scalar(0, 0, 255), 1);
          detections.push_back(bb);
        }
      }

      return detections;
    }

    virtual void SetProperty(const std::string& key, const std::string& value) override
    {
      CSubject<uint8_t, uint8_t>::SetProperty(key, value);
    }

  protected:

    cv::Ptr<cv::BackgroundSubtractor> pBackgroundSubtractor = nullptr;

};

#endif