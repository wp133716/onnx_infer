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
    // const char* model_path = "../network/network_track/model_test.onnx";
    // const char* model_path = "../tf_model_10.onnx";
    const char* model_path = "../test.onnx";


    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    // GetOutputCount
    size_t num_output_nodes = session.GetOutputCount();
    std::cout << "Number of outputs = " << num_output_nodes << "\n";

    // GetInputName
    std::vector<const char*> input_node_names(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        char* input_name = session.GetInputName(i, allocator);
        input_node_names[i] = input_name;
    }
    // print input name
    std::cout << "Number of inputs = " << num_input_nodes << "\n";
    for (int i = 0; i < num_input_nodes; i++) {
        std::cout << "Input " << i << " : " << input_node_names[i] << "\n";
    }
    std::vector<const char*> output_node_names(num_output_nodes);
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }
    // std::vector<int64_t> input_node_dims = {1, 50, 9};
    // std::vector<int64_t> input_node_dims2 = {1, 50, 2};
    std::vector<std::vector<int64_t>> input_node_dims(num_input_nodes);
    std::vector<const char*> input_type(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        auto input_type_info = session.GetInputTypeInfo(i);
        auto input_tensor_type_info = input_type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = input_tensor_type_info.GetElementType();
        if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT){
            std::cout<<"float"<<std::endl;
            // input_type[i] = float;
            input_type[i] = "float";
            
        }
        else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE){
            std::cout<<"double"<<std::endl;
            input_type[i] = "double";
        }
        input_node_dims[i] = input_tensor_type_info.GetShape();
        std::cout << "shape=[";
        for (int j = 0; j < input_node_dims[i].size(); j++) {
            std::cout << input_node_dims[i][j] << ",";
        }
        std::cout << "]\n";
        
    }

    // 设置输入
    size_t input_tensor_size = input_node_dims[0][1] * input_node_dims[0][2]; //50*9
    input_node_dims[0][0] = 1;
    std::cout<<"input_tensor_size :"<<input_tensor_size<<std::endl;
    // input2
    // size_t input_tensor_size2 = input_node_dims[1][1] * input_node_dims[1][2]; // 50 * 2;
    std::vector<double> input_tensor_values(input_tensor_size);
    // std::vector<double> input_tensor_values2(input_tensor_size2);


    //测试100次所需时间
    auto start = std::chrono::system_clock::now();
    for(int i=0; i<1; i++)
    {   
        // 测试每次所需时间
        auto start2 = std::chrono::system_clock::now();
        for (unsigned int i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = float(i)/100; //float(virtual_image[i]);
        // for (unsigned int i = 0; i < input_tensor_size2; i++)
        //     input_tensor_values2[i] = 1.f; //float(virtual_image[i]);
        // create input tensor object from data values ！！！！！！！！！！
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // auto memory_info2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<double>(memory_info, input_tensor_values.data(),
                                                                input_tensor_size, input_node_dims[0].data(), 3);
        // Ort::Value input_tensor2 = Ort::Value::CreateTensor<double>(memory_info2, input_tensor_values2.data(),
        //                                                         input_tensor_size2, input_node_dims[1].data(), 3);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        // ort_inputs.push_back(std::move(input_tensor2));
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                        ort_inputs.size(), output_node_names.data(), 1);

        auto outputInfo = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputInfo.GetShape();
        std::cout<<"output shape : ["<<outputShape[0]<<","<<outputShape[1]<<","<<outputShape[2]<<"]"<<std::endl;
        // std::cout<<output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1]<<std::endl;
        // 从output_tensors中获取输出：
        // int64_t num_elements = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        // std::cout<<num_elements<<std::endl;

        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        for (int i=0; i<outputShape[0]; i++)
            for(int j=0; j<outputShape[1]; j++)
            {
                for(int k=0; k<outputShape[2]; k++)
                    std::cout<<floatarr[i*outputShape[1]*outputShape[2]+j*outputShape[2]+k]<<" ";
                std::cout<<std::endl;
            }



        // float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        // for (int i=0; i<num_output_nodes; i++)
        // {
        //     std::cout<<floatarr[i]<<std::endl;
        // }

        // output_tensors[0].GetTensorTypeAndShapeInfo
        auto end2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds2 = end2-start2;
        std::cout << "elapsed time: " << elapsed_seconds2.count() << "s\n";
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}

// input_tensor_size :400
// output shape : [1,10,3]
// 3.10425 3.50898 2.48649 
// 3.07826 3.44839 2.43524 
// 3.0685 3.40899 2.38765 
// 3.05186 3.36237 2.34051 
// 3.02755 3.30252 2.28578 
// 3.00407 3.24827 2.23652 
// 2.99455 3.20999 2.19042 
// 2.96438 3.14574 2.14243 
// 2.95098 3.10209 2.09512 
// 2.93998 3.0599 2.04294 
// elapsed time: 0.0255257s