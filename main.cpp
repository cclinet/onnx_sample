#include "incbin/incbin.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <optional>

class OnnxModel {
public:
  OnnxModel(const void *model, const size_t size)
      : session_(env, model, size, session_options_) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  }
  void run(std::array<float, 97 * 8> &input,
           std::array<float, 97 * 2> &output) {
    const char *input_names[] = {"l_tensor_x_"};
    const char *output_names[] = {"fc3_1"};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    input_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_shape_.data(),
        input_shape_.size());

    output_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info, output.data(), output.size(), output_shape_.data(),
        output_shape_.size());

    Ort::RunOptions run_options;
    session_.Run(run_options, input_names, &input_tensor_, 1, output_names,
                 &output_tensor_, 1);
    std::cout << output_tensor_ << std::endl;
  }

private:
  Ort::Env env;
  Ort::SessionOptions session_options_{};
  Ort::Session session_;

  Ort::Value input_tensor_{nullptr};
  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> input_shape_ = {97, 8};
  std::array<int64_t, 2> output_shape_{97, 2};
};

extern "C" {
#ifdef MODEL_PATH
INCBIN(Onnx, MODEL_PATH);
#endif
}
int main(int, char **) {
  std::cout << "Hello, from yolo!\n";
  std::cout << "Icon size: " << gOnnxSize << std::endl;
  auto model = OnnxModel(gOnnxData, gOnnxSize);
  std::array<float, 97 * 8> input{};
  input.fill(1.0f);
  std::array<float, 97 * 2> output{};
  model.run(input, output);
  for (auto &o : output) {
    std::cout << o << " ";
  }
}
