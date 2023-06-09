#ifndef ONNX_INFER_H
#define ONNX_INFER_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

using namespace std;

class OnnxInfer {
public:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    OnnxInfer(const string& model_path);
    ~OnnxInfer();
    vector<float> forward(vector<float> input_data_0, vector<float> input_data_1);
    vector<float> forward(vector<double> input_data_0, vector<double> input_data_1);
// private:
    // Ort::Env env;
    // Ort::SessionOptions session_options;
    // // const char* onnxModelPath;
    // Ort::Session session;

    vector<const char*> input_node_names;
    vector<const char*> output_node_names;
    vector<vector<int64_t>> input_node_dims;
    vector<const char*> input_type;

};

#endif // ONNX_INFER_H