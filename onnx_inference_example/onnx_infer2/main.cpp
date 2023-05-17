#include "onnx_infer.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace std;


int main() {
    // 设置输入
    vector<double> input_data_0;
    vector<double> input_data_1;
    input_data_0.resize(1 * 50 * 9);
    input_data_1.resize(1 * 50 * 2);
    for (int i = 0; i < 1 * 50 * 9; i++) {
        input_data_0[i] = 1.l; //1.f是float类型，1是double类型
    }
    for (int i = 0; i < 1 * 50 * 2; i++) {
        input_data_1[i] = 1.l;
    }
    // 推理,获取输出
    const string model_path = "../model_test.onnx";
    OnnxInfer onnx_infer(model_path);

    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < 1; i++)
    {
        vector<float> output_data;
        // onnx_infer.input_type[0]
        std::cout<<"onnx_infer.input_type : "<<onnx_infer.input_type[0]<<std::endl; 
        //判断输入类型是否为float
        
        if (strcmp(onnx_infer.input_type[0], "float") == 0){
        // if (onnx_infer.input_type[0] == "float"){
            //将input_data_0和input_data_1转换为float类型
            vector<float> input_data_0_float(input_data_0.begin(), input_data_0.end());
            vector<float> input_data_1_float(input_data_1.begin(), input_data_1.end());
            output_data = onnx_infer.forward(input_data_0_float, input_data_1_float);
        }
        else if (strcmp(onnx_infer.input_type[0], "double") == 0)
            output_data = onnx_infer.forward(input_data_0, input_data_1);
        else
            cout << "input_type error" << endl;
        // 输出结果
        for (int i = 0; i < 4; i++) {
            cout << output_data[i] << " ";
        }
        cout << endl;
        
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}

