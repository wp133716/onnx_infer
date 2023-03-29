#include "onnx_infer.h"

OnnxInfer::OnnxInfer(const string& model_path) : session(env, model_path.c_str(), session_options) {
    // 初始化环境
    // env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // OrtCUDAProviderOptions cuda_options{
    //   0,
    //   OrtCudnnConvAlgoSearch::EXHAUSTIVE,
    //   std::numeric_limits<size_t>::max(),
    //   0,
    //   true
    // };

    // 初始化session
    // session_options.AppendExecutionProvider_CUDA(cuda_options);

    // Ort::Session session(env, model_path.c_str(), session_options);

    input_node_names = {"input_0", "input_1"};
    output_node_names = {"dense_2", "tf.math.multiply_2"};
    input_node_dims = {{1, 50, 9}, {1, 50, 2}};
}

OnnxInfer::~OnnxInfer()
{

}

vector<float> OnnxInfer::forward(vector<float> input_data_0, vector<float> input_data_1) {
    // static Ort::Session session(env, onnxModelPath, session_options);
    std::vector<Ort::Value> ort_inputs;

    // 设置输入
    Ort::MemoryInfo memory_info_0 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo memory_info_1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_0 = Ort::Value::CreateTensor<float>(memory_info_0, input_data_0.data(),
                                                                input_data_0.size(), input_node_dims[0].data(), input_node_dims[0].size());
    Ort::Value input_tensor_1 = Ort::Value::CreateTensor<float>(memory_info_1, input_data_1.data(),
                                                                input_data_1.size(), input_node_dims[1].data(), input_node_dims[1].size());
    ort_inputs.push_back(std::move(input_tensor_0));
    ort_inputs.push_back(std::move(input_tensor_1));
    // 推理
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                input_node_names.data(),
                                ort_inputs.data(),
                                ort_inputs.size(),
                                output_node_names.data(),
                                output_node_names.size());
    // 获取输出
    vector<float> output_data;
    output_data.resize(output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
    memcpy(output_data.data(), output_tensors[0].GetTensorMutableData<float>(), output_data.size() * sizeof(float));
    
    return output_data;
}