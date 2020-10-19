#include <functional>
#include <iostream>
#include <vector>
#include <string>

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat&);

int main(int argc, char *argv[])
{
  InferenceEngine::Core core;

  auto network = core.ReadNetwork(argv[1]);

  /** Take information about all topology inputs **/
  InferenceEngine::InputsDataMap input_info = network.getInputsInfo();
  /** Take information about all topology outputs **/
  InferenceEngine::OutputsDataMap output_info = network.getOutputsInfo();

  /** Iterate over all input info**/
  for (auto &item : input_info)
  {
    auto input_data = item.second;
    input_data->setPrecision(InferenceEngine::Precision::U8);
    input_data->setLayout(InferenceEngine::Layout::NCHW);
    input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
    input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
  }

  /** Iterate over all output info**/
  for (auto &item : output_info)
  {
    auto output_data = item.second;
    output_data->setPrecision(InferenceEngine::Precision::FP32);
  }
  InferenceEngine::DataPtr& _output = output_info.begin()->second;
  const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
  auto max_detections_count_ = outputDims[2];
  auto object_size_ = outputDims[3];

  auto executable_network = core.LoadNetwork(network, "CPU");

  auto req = executable_network.CreateInferRequest();

  /*
   * Read input image to a blob and set it to an infer 
   * request without resize and layout conversions.
   */

  cv::Mat image = cv::imread(argv[2]);
  auto width_ = static_cast<float>(image.cols);
  auto height_ = static_cast<float>(image.rows);

  InferenceEngine::Blob::Ptr blob = wrapMat2Blob(image);  // just wrap Mat data by Blob::Ptr without allocating of new memory

  req.SetBlob(input_info.begin()->first, blob);  // infer_request accepts input blob of any size

  req.Infer();

  InferenceEngine::Blob::Ptr output = req.GetBlob(output_info.begin()->first);

  InferenceEngine::LockedMemory<const void> outputMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(output)->rmap();

  const float *data = outputMapped.as<float *>();

    for (int det_id = 0; det_id < max_detections_count_; ++det_id) 
	{
        const int start_pos = det_id * object_size_;

        const float batchID = data[start_pos];
        if (batchID == -1.0 ) {
            break;
        }

        const float score = std::min(std::max(0.0f, data[start_pos + 2]), 1.0f);
        const float x0 =
            std::min(std::max(0.0f, data[start_pos + 3]), 1.0f) * width_;
        const float y0 =
            std::min(std::max(0.0f, data[start_pos + 4]), 1.0f) * height_;
        const float x1 =
            std::min(std::max(0.0f, data[start_pos + 5]), 1.0f) * width_;
        const float y1 =
            std::min(std::max(0.0f, data[start_pos + 6]), 1.0f) * height_;

        auto rect = cv::Rect(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                         static_cast<int>(round(static_cast<double>(y0)))),
                               cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                         static_cast<int>(round(static_cast<double>(y1)))));

          cv::rectangle(image, rect, 
               cv::Scalar(255, 0, 0 ), 1, 1);  // detection blue
    }

    cv::imshow("ddd", image);

	cv::waitKey(0);

  return 0;
}

/**
 * @brief Wraps data stored inside of a passed cv::Mat object by new Blob pointer.
 * @note: No memory allocation is happened. The blob just points to already existing
 *        cv::Mat data.
 * @param mat - given cv::Mat object with an image data.
 * @return resulting Blob pointer.
 */
InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat) {
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