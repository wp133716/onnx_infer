#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

using namespace std;


int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    // OrtCUDAProviderOptions cuda_options{
    //       0,
    //       OrtCudnnConvAlgoSearch::EXHAUSTIVE,
    //       std::numeric_limits<size_t>::max(),
    //       0,
    //       true
    //   };

    // session_options.AppendExecutionProvider_CUDA(cuda_options);
    // const char* model_path = "./overlapTransformer.onnx";
    const char* model_path = "../model_test.onnx";


    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();

    // 测试两个输入的网络
    std::vector<const char*> input_node_names = {"input_0", "input_1"};
    std::vector<const char*> output_node_names = {"dense_2", "tf.math.multiply_2"};

    std::vector<int64_t> input_node_dims = {1, 50, 9};
    std::vector<int64_t> input_node_dims2 = {1, 50, 2};
    
    // 设置输入
    size_t input_tensor_size = 50 * 9;
    // input2
    size_t input_tensor_size2 = 50 * 2;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<float> input_tensor_values2(input_tensor_size2);

    //测试100次所需时间
    auto start = std::chrono::system_clock::now();
    for(int i=0; i<100; i++)
    {   
        // 测试每次所需时间
        auto start2 = std::chrono::system_clock::now();
        for (unsigned int i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = 1.f; //float(virtual_image[i]);
        for (unsigned int i = 0; i < input_tensor_size2; i++)
            input_tensor_values2[i] = 1.f; //float(virtual_image[i]);
        // create input tensor object from data values ！！！！！！！！！！
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto memory_info2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                                input_tensor_size, input_node_dims.data(), 3);
        Ort::Value input_tensor2 = Ort::Value::CreateTensor<float>(memory_info2, input_tensor_values2.data(),
                                                                input_tensor_size2, input_node_dims2.data(), 3);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        ort_inputs.push_back(std::move(input_tensor2));
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                        ort_inputs.size(), output_node_names.data(), 1);
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        for (int i=0; i<4; i++)
        {
            std::cout<<floatarr[i]<<std::endl;
        }
        auto end2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds2 = end2-start2;
        std::cout << "elapsed time: " << elapsed_seconds2.count() << "s\n";
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
