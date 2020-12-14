#ifndef DETECTOR_HPP
#define DETECTOR_HPP 

#include <tuple>
#include <string>
#include <filesystem>

#include <opencv2/dnn.hpp>
#include <opencv2/bgsegm.hpp>
#include <inference_engine.hpp>

#include <Geometry.hpp>

#include <CSubject.hpp>

class CDetector : public NPL::CSubject<uint8_t, uint8_t>
{
  public:

    CDetector() {}

    CDetector(const std::string& config, const std::string& weight)
    {
      if (getenv("cpp-cvl-test"))
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
		    iNetwork.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CPU);
		    std::cout << "IE backend and cpu target enabled for inference." << std::endl;
	    }
    }

    virtual ~CDetector() {}

    virtual Detections Detect(cv::Mat& frame) = 0;

    std::string TrailToPath(TrackingContext& tc)
    {
      std::string path;

      for (auto& r : tc.iTrail)
      {
        auto p = GetRectCenter(r);

        if (path.size())
        {
          path += ", ";
        }

        path += std::to_string(p.x) + " " + std::to_string(p.y);
      }

      return path;
    }

  protected:

    std::string iTarget;

    std::string iConfigFile;

    std::string iWeightFile;

    cv::dnn::Net iNetwork;
};

using SPCDetector = std::shared_ptr<CDetector>;

class AgeGenderDetector : public CDetector
{
  public:

    AgeGenderDetector() : CDetector(
        "age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml",
        "age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin")
    {
    }

    virtual Detections Detect(cv::Mat& frame) override
    {
      auto blob = cv::dnn::blobFromImage(frame, 1, cv::Size(62, 62));
      iNetwork.setInput(blob);
      std::vector<cv::Mat> out;
      std::vector<cv::String> layers = iNetwork.getUnconnectedOutLayersNames();
      iNetwork.forward(out, layers);
      return {
        {
          cv::Rect2d(), 
          out[0].at<float>(0) * 100,
          out[1].at<float>(1),
          false
        }
      };
    }

  protected:

};

using SPAgeGenderDetector = std::shared_ptr<AgeGenderDetector>;

class FaceDetector : public CDetector
{
  public:

    FaceDetector() : CDetector(
        "face-detection-retail-0005/FP16/face-detection-retail-0005.xml", 
        "face-detection-retail-0005/FP16/face-detection-retail-0005.bin")
    {
      iAgeGenderDetector = std::make_shared<AgeGenderDetector>();
    }

    ~FaceDetector() {}

    virtual Detections Detect(cv::Mat& frame) override
    {
      Detections out;

      cv::Mat inputBlob = cv::dnn::blobFromImage(
        frame,
        1.0f,
        cv::Size(300, 300), //672, 384), //
        cv::Scalar(104.0, 177.0, 123.0),
        false,
        false);

      iNetwork.setInput(inputBlob);

      cv::Mat detection = iNetwork.forward();

      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

      for (int i = 0; i < detectionMat.rows; ++i)
      {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > 0.8)
        {
          int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
          int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
          int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
          int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

          auto rect = cv::Rect2d(x1, y1, x2-x1, y2-y1);

          if (IsRectInsideMat(rect, frame))
          {
            cv::Mat roi = frame(rect);

            Detections ag;

            if (iAgeGenderDetector)
            {
              ag = iAgeGenderDetector->Detect(roi);
            }

            out.emplace_back(rect, std::get<1>(ag[0]), std::get<2>(ag[0]), false);
          }
        }
      }

      return out;
    }

    virtual void OnEvent(std::any e)
    {
      auto tc = std::any_cast<std::reference_wrapper<TrackingContext>>(e).get();

      if (tc.iThumbnails.size())
      {
        auto out = std::make_tuple(
           std::string(),
           std::string(),
           std::vector<uchar>(),
           std::ref(tc));

        auto& path = std::get<0>(out);
        auto& demography = std::get<1>(out);
        auto& thumb = std::get<2>(out);

        path = TrailToPath(tc);

        if (tc.iAge.size() != tc.iGender.size()) 
        {
          throw std::exception("age-gender size mismatch");
        }

        for (int i = 0; i < tc.iAge.size(); ++i)
        {
          if (demography.size())
          {
            demography += ", ";
          }

          demography += std::to_string((int)tc.iAge[i]) + " " + tc.iGender[i];         
        }

        if (demography.size())
        {
          demography += std::string(", ") + std::to_string(tc.getAge()) + std::string(" ") + (tc.isMale() ? std::string("M") : std::string("F"));
        }

        cv::imencode(".jpg", tc.iThumbnails[tc.iThumbnails.size() / 2], thumb);

        CSubject<uint8_t, uint8_t>::OnEvent(std::ref(out));
      }
    }

  protected:

    SPAgeGenderDetector iAgeGenderDetector = nullptr;
};

class PeopleDetector : public CDetector
{
  public:

    PeopleDetector() : CDetector(
        "person-detection-retail-0013/FP16/person-detection-retail-0013.xml", 
        "person-detection-retail-0013/FP16/person-detection-retail-0013.bin")
    {
      //person-detection-retail-0013 
      //pedestrian-detection-adas-0002
    }

    virtual Detections Detect(cv::Mat& frame) override
    {
      Detections out;

      cv::Mat inputBlob = cv::dnn::blobFromImage(
        frame,
        1,
        cv::Size(544, 320)); /*, //300, 300
        cv::Scalar(),
        false,
        false,
        CV_8U);*/

      iNetwork.setInput(inputBlob);

      cv::Mat detection = iNetwork.forward();

      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

      for (int i = 0; i < detectionMat.rows; ++i)
      {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > 0.7)
        {
          int x1, y1, x2, y2;

          x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
          y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
          x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
          y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

          out.emplace_back(
            cv::Rect2d(cv::Point(x1, y1), cv::Point(x2, y2)), -1.0f, -1.0f, false
          );
        }
      }

      return out;
    }

    virtual void OnEvent(std::any e)
    {
      auto tc = std::any_cast<std::reference_wrapper<TrackingContext>>(e).get();

      if (tc.iThumbnails.size())
      {
        auto out = std::make_tuple(
           std::string(),
           std::string(),
           std::vector<uchar>(),
           std::ref(tc));

        auto& path = std::get<0>(out);
        auto& thumb = std::get<2>(out);

        path = TrailToPath(tc);

        cv::imencode(".jpg", tc.iThumbnails[tc.iThumbnails.size() / 2], thumb);

        CSubject<uint8_t, uint8_t>::OnEvent(std::ref(out));
      }
    }    

  protected:

};

class ObjectDetector : public CDetector
{
  public:

    ObjectDetector(const std::string& target) :
     CDetector("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel") 
    {
      iTarget = target;
    }

    virtual Detections Detect(cv::Mat& frame) override
    {
      Detections out;

      cv::Mat inputBlob = cv::dnn::blobFromImage(
        frame,
        0.007843f,
        cv::Size(300, 300),
        cv::Scalar(127.5, 127.5, 127.5),
        false, 
        false);

      iNetwork.setInput(inputBlob);

      cv::Mat detection = iNetwork.forward();

      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

      for (int i = 0; i < detectionMat.rows; ++i)
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

            out.emplace_back(
               cv::Rect2d(cv::Point(x1, y1), cv::Point(x2, y2)), -1.0f, -1.0f, false
            );
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

    virtual Detections Detect(cv::Mat& frame) override
    {
      cv::Mat fgMask;

      std::vector<
        std::vector<cv::Point>
      > contours;

      Detections out;

      pBackgroundSubtractor->apply(frame, fgMask, 0.8);

      cv::findContours(fgMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

      auto areaThreshold = GetPropertyAsInt("bbarea");
      auto excludeHBB = GetPropertyAsInt("exhzbb");

      for (size_t i = 0; i < contours.size(); ++i) 
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

        for (size_t j = 0; j < contours.size(); ++j)
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
          out.emplace_back(bb, -1.0f, -1.0f, false);
        }
      }

      return out;
    }

    virtual void SetProperty(const std::string& key, const std::string& value) override
    {
      CSubject<uint8_t, uint8_t>::SetProperty(key, value);
    }

  protected:

    cv::Ptr<cv::BackgroundSubtractor> pBackgroundSubtractor = nullptr;

};

class IEDetector : public CDetector
{
  public:

    IEDetector(const std::string& target)
    {
      InferenceEngine::CNNNetwork network;

      iTarget = target;

      if (iTarget == "people")
      {
        network = iCore.ReadNetwork("../../cpp-cvl/MODELS/person-detection-retail-0013/FP16/person-detection-retail-0013.xml");
      }
      else if (iTarget == "face")
      {
         network = iCore.ReadNetwork("../../cpp-cvl/MODELS/face-detection-retail-0005/FP16/face-detection-retail-0005.xml");

         iAgeGenderDetector = std::make_shared<AgeGenderDetector>();
      }

      iInputDataMap = network.getInputsInfo();

      iOutputDataMap = network.getOutputsInfo();

      for (auto &item : iInputDataMap)
      {
        auto input_data = item.second;
        input_data->setPrecision(InferenceEngine::Precision::U8);
        input_data->setLayout(InferenceEngine::Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::BGR);
      }

      for (auto &item : iOutputDataMap)
      {
        auto output_data = item.second;
        output_data->setPrecision(InferenceEngine::Precision::FP32);
      }

      InferenceEngine::DataPtr& _output = iOutputDataMap.begin()->second;

      const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();

      iMaxDetections = outputDims[2];
      iObjectSize = outputDims[3];

      iNetwork = iCore.LoadNetwork(network, "CPU");
    }

    virtual Detections Detect(cv::Mat& frame) override
    {
      Detections out;

      auto req = iNetwork.CreateInferRequest();

      auto width_ = static_cast<float>(frame.cols);
      auto height_ = static_cast<float>(frame.rows);

      InferenceEngine::Blob::Ptr blob = wrapMat2Blob(frame);

      req.SetBlob(iInputDataMap.begin()->first, blob); 

      req.Infer();

      InferenceEngine::Blob::Ptr output = req.GetBlob(iOutputDataMap.begin()->first);

      auto outputMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(output)->rmap();

      const float *data = outputMapped.as<float *>();

      for (int i = 0; i < iMaxDetections; ++i) 
	    {
        const int start_pos = i * iObjectSize;

        const float score = std::min(std::max(0.0f, data[start_pos + 2]), 1.0f);

        const float x0 = std::min(std::max(0.0f, data[start_pos + 3]), 1.0f) * width_;
        const float y0 = std::min(std::max(0.0f, data[start_pos + 4]), 1.0f) * height_;
        const float x1 = std::min(std::max(0.0f, data[start_pos + 5]), 1.0f) * width_;
        const float y1 = std::min(std::max(0.0f, data[start_pos + 6]), 1.0f) * height_;

        auto rect = cv::Rect(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                       static_cast<int>(round(static_cast<double>(y0)))),
                             cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                       static_cast<int>(round(static_cast<double>(y1)))));

        if (rect.area() > 0 && score >= 0.7)
        {
          if (iTarget == "face")
          {
            cv::Mat roi = frame(rect);

            Detections ag;

            if (iAgeGenderDetector)
            {
              ag = iAgeGenderDetector->Detect(roi);
            }

            out.emplace_back(rect, std::get<1>(ag[0]), std::get<2>(ag[0]), false);
          }
          else
          {
            out.emplace_back(rect, -1.0f, -1.0f, false);
          }
        }
      }

      return out;
    }

  protected:

    InferenceEngine::Core iCore;

    InferenceEngine::ExecutableNetwork iNetwork;

    InferenceEngine::InputsDataMap iInputDataMap;

    InferenceEngine::OutputsDataMap iOutputDataMap;

    int iMaxDetections;

    int iObjectSize;

    std::string iTarget;

    SPAgeGenderDetector iAgeGenderDetector = nullptr;

    InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat) 
    {
      size_t channels = mat.channels();
      size_t height = mat.size().height;
      size_t width = mat.size().width;

      size_t strideH = mat.step.buf[0];
      size_t strideW = mat.step.buf[1];

      bool is_dense =
            strideW == channels &&
            strideH == channels * width;

      if (!is_dense) THROW_IE_EXCEPTION
                << "Doesn't support conversion from not dense cv::Mat";

      InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
                                      {1, channels, height, width},
                                      InferenceEngine::Layout::NHWC);

      return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
    }
};

void FilterDetections(Detections& detections, cv::Mat& m)
{
  for (auto& it = detections.begin(); it != detections.end(); )
  {
    bool remove = false;

    auto& roi = std::get<0>(*it);

    if (roi.x + roi.width > m.cols || roi.y + roi.height > m.rows)
    {
      remove = true;
    }

    //exclude near-to-frame detections, mark white
    if ((roi.y < 5) || ((roi.y + roi.height) > (m.rows - 5)))
    {
      cv::rectangle(m, roi, cv::Scalar(255, 255, 255), 1, 1); 
      remove =true;
    }

    if (remove)
    {
      it = detections.erase(it);
    }
    else
    {
      it++;
    }
  }
}

#endif