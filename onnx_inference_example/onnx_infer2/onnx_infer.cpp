#include "onnx_infer.h"

// OnnxInfer::OnnxInfer() {
//     cout << "无参构造OnnxInfer::OnnxInfer" << endl;
// }


OnnxInfer::OnnxInfer(const string& model_path) : session(env, model_path.c_str(), session_options) {
    cout << "有参构造OnnxInfer::OnnxInfer" << endl;

    // input_node_names = {"input_0", "input_1"};
    // output_node_names = {"dense_2", "tf.math.multiply_2"};
    // input_node_dims = {{1, 50, 9}, {1, 50, 2}};

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::cout << "Number of inputs = " << num_input_nodes << "\n";
    // GetOutputCount
    size_t num_output_nodes = session.GetOutputCount();
    std::cout << "Number of outputs = " << num_output_nodes << "\n";

    // GetInputName
    input_node_names.resize(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        char* input_name = session.GetInputName(i, allocator);
        input_node_names[i] = input_name;
    }
    // GetOutputName
    output_node_names.resize(num_output_nodes);
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        output_node_names[i] = output_name;
    }
    input_node_dims.resize(num_input_nodes);
    input_type.resize(num_input_nodes);
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        std::cout << "Input " << i << " name : " << input_node_names[i] << "\n";
        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto type = tensor_info.GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            input_type[i] = "float";
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE){
            input_type[i] = "double";
        }

        std::cout << "Input " << i << " type: " << input_type[i] << "\n";
        // print input shapes/dims
        input_node_dims[i] = tensor_info.GetShape();
        input_node_dims[i][0] = 1;
        std::cout << "Input " << i << " shape: ";
        for (int j = 0; j < input_node_dims[i].size(); j++) {
            std::cout << input_node_dims[i][j] << " ";
        }
        std::cout << "\n";
    }
    
}

OnnxInfer::~OnnxInfer()
{
    cout << "OnnxInfer::~OnnxInfer" << endl;
}

vector<float> OnnxInfer::forward(vector<float> input_data_0, vector<float> input_data_1) {
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

    // // print input_node_dims
    // std::cout<<"------------forward------------"<<std::endl;
    // for (int i = 0; i < input_node_dims.size(); i++) {
    //     std::cout << "Input " << i << " shape: ";
    //     for (int j = 0; j < input_node_dims[i].size(); j++) {
    //         std::cout << input_node_dims[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }

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

vector<float> OnnxInfer::forward(vector<double> input_data_0, vector<double> input_data_1) {
    std::vector<Ort::Value> ort_inputs;

    // 设置输入
    Ort::MemoryInfo memory_info_0 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo memory_info_1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_0 = Ort::Value::CreateTensor<double>(memory_info_0, input_data_0.data(),
                                                                input_data_0.size(), input_node_dims[0].data(), input_node_dims[0].size());
    Ort::Value input_tensor_1 = Ort::Value::CreateTensor<double>(memory_info_1, input_data_1.data(),
                                                                input_data_1.size(), input_node_dims[1].data(), input_node_dims[1].size());

    ort_inputs.push_back(std::move(input_tensor_0));
    ort_inputs.push_back(std::move(input_tensor_1));
    // 推理

    // // print input_node_dims
    // for (int i = 0; i < input_node_dims.size(); i++) {
    //     std::cout << "Input " << i << " shape: ";
    //     for (int j = 0; j < input_node_dims[i].size(); j++) {
    //         std::cout << input_node_dims[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }

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