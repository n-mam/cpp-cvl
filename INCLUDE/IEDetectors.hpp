
# pragma once

#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>

#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

/**
* @brief Sets image data stored in cv::Mat object to a given Blob object.
* @param orig_image - given cv::Mat object with an image data.
* @param blob - Blob object which to be filled by an image data.
* @param batchIndex - batch index of an image inside of the blob.
*/
template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    if (static_cast<size_t>(orig_image.channels()) != channels) {
        THROW_IE_EXCEPTION << "The number of channels for net input and image must match";
    }
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    T* blob_data = blobMapped.as<T*>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
            static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    if (channels == 1) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
            }
        }
    } else if (channels == 3) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t  h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] =
                            resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported number of channels";
    }
}

struct BaseDetection 
{
    InferenceEngine::ExecutableNetwork net;
    InferenceEngine::InferRequest::Ptr request;
    std::string topoName;
    std::string pathToModel;
    std::string deviceForInference;
    const size_t maxBatch;
    bool isBatchDynamic;
    const bool isAsync;
    mutable bool enablingChecked;
    mutable bool _enabled;
    const bool doRawOutputMessages;

    BaseDetection(const std::string &topoName,
                  const std::string &pathToModel,
                  const std::string &deviceForInference,
                  int maxBatch, bool isBatchDynamic, bool isAsync,
                  bool doRawOutputMessages)
                  : topoName(topoName), 
                    pathToModel(pathToModel), 
                    deviceForInference(deviceForInference),
                    maxBatch(maxBatch), 
                    isBatchDynamic(isBatchDynamic), 
                    isAsync(isAsync),
                    enablingChecked(false),
                    _enabled(false), 
                    doRawOutputMessages(doRawOutputMessages)
    {

    }

    virtual ~BaseDetection() {}

    InferenceEngine::ExecutableNetwork* operator ->()
    {
      return &net;
    }

    virtual InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) = 0;

    virtual void submitRequest()
    {
      if (!enabled() || request == nullptr) return;
      if (isAsync) {
        request->StartAsync();
      } else {
        request->Infer();
      }
    }

    virtual void wait()
    {
      if (!enabled()|| !request || !isAsync)
        return;
      request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    
    bool enabled() const
    {
      if (!enablingChecked)
      {
        _enabled = !pathToModel.empty();
        enablingChecked = true;
      }
      
      return _enabled;
    }

};

struct AgeGenderDetection : BaseDetection
{
    struct Result 
    {
      float age;
      float maleProb;
    };

    std::string input;
    std::string outputAge;
    std::string outputGender;
    size_t enquedFaces;

    AgeGenderDetection(const std::string &pathToModel,
                       const std::string &deviceForInference,
                       int maxBatch, bool isBatchDynamic, bool isAsync,
                       bool doRawOutputMessages) 
                    : BaseDetection("Age/Gender", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
                      enquedFaces(0)
    {

    }

    InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) override
    {
      // Read network
      auto network = ie.ReadNetwork(pathToModel);
      // Set maximum batch size to be used.
      network.setBatchSize(maxBatch);

      // ---------------------------Check inputs -------------------------------------------------------------
      // Age/Gender Recognition network should have one input and two outputs

      InputsDataMap inputInfo(network.getInputsInfo());
      if (inputInfo.size() != 1) {
        throw std::logic_error("Age/Gender Recognition network should have only one input");
      }
      InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
      inputInfoFirst->setPrecision(Precision::U8);
      input = inputInfo.begin()->first;
      // -----------------------------------------------------------------------------------------------------

      // ---------------------------Check outputs ------------------------------------------------------------

      OutputsDataMap outputInfo(network.getOutputsInfo());
      if (outputInfo.size() != 2) {
        throw std::logic_error("Age/Gender Recognition network should have two output layers");
      }
      auto it = outputInfo.begin();

      DataPtr ptrAgeOutput = (it++)->second;
      DataPtr ptrGenderOutput = (it++)->second;

      outputAge = ptrAgeOutput->getName();
      outputGender = ptrGenderOutput->getName();

      _enabled = true;
      return network;
    }

    void submitRequest() override
    {
      if (!enquedFaces)
        return;
      if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
      }
      BaseDetection::submitRequest();
      enquedFaces = 0;       
    }

    void enqueue(const cv::Mat &face)
    {
      if (!enabled()) {
        return;
      }

      if (enquedFaces == maxBatch) {
        std::cout << "Number of detected faces more than maximum(" << maxBatch << ") processed by Age/Gender Recognition network\n";
        return;
      }

      if (!request) {
        request = net.CreateInferRequestPtr();
      }

      Blob::Ptr  inputBlob = request->GetBlob(input);

      matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

      enquedFaces++;
    }

    Result operator[] (int idx) const
    {
      Blob::Ptr  genderBlob = request->GetBlob(outputGender);
      Blob::Ptr  ageBlob    = request->GetBlob(outputAge);

      LockedMemory<const void> ageBlobMapped = as<MemoryBlob>(ageBlob)->rmap();
      LockedMemory<const void> genderBlobMapped = as<MemoryBlob>(genderBlob)->rmap();

      AgeGenderDetection::Result r = {ageBlobMapped.as<float*>()[idx] * 100,
                                      genderBlobMapped.as<float*>()[idx * 2 + 1]};
      if (doRawOutputMessages) {
        std::cout << "[" << idx << "] element, male prob = " << r.maleProb << ", age = " << r.age << std::endl;
      }

      return r;        
    }
};