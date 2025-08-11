//
// Created by manu on 2025/7/10.
//

#include <onnxruntime_cxx_api.h>
#include <fftw3.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cassert>
#include <algorithm>   // sort
#include <numeric>     // accumulate

int main() {
    /* 1. 创建 Env + Session */
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "demo"};
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    const char *model_path = "/home/manu/tmp/ecg_net_model.onnx";
    Ort::Session session{env, model_path, opts};

    /* ------------------ input0: signal ------------------ */
    std::vector<float> signal_values(1 * 3 * 256, 0.f);
    std::array<int64_t, 3> signal_shape = {1, 3, 256};

    /* ------------------ input1: enhance ----------------- */
    std::array<float, 2> enhance_values = {0.f, 0.f};         // 随便先填 0
    std::array<int64_t, 2> enhance_shape = {1, 2};

    /* 2. 把数据包装成 Ort::Value */
    Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value signal_tensor =
            Ort::Value::CreateTensor<float>(mem_info,
                                            signal_values.data(), signal_values.size(),
                                            signal_shape.data(), signal_shape.size());

    Ort::Value enhance_tensor =                                   // ★ 新增
            Ort::Value::CreateTensor<float>(mem_info,
                                            enhance_values.data(), enhance_values.size(),
                                            enhance_shape.data(), enhance_shape.size());

    /* 3. 调用 Run */
    const char *input_names[] = {"signal", "enhance"};           // ★ 两个输入
    const char *output_names[] = {"output"};

    Ort::Value input_tensors[] = {std::move(signal_tensor),
                                  std::move(enhance_tensor)};    // ★ 两个张量

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, input_tensors, 2,   // ★ input 个数=2
                                      output_names, 1);

    /* 4. 取出并打印结果 */
    float *out = output_tensors[0].GetTensorMutableData<float>();

    size_t out_elem =
            output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "Output: ";
    for (size_t i = 0; i < out_elem; ++i)
        std::cout << out[i] << ' ';
    std::cout << '\n';

    return 0;
}
