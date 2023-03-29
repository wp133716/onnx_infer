#include "onnx_infer.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <string>

using namespace std;


int main() {
    // 设置输入
    vector<float> input_data_0;
    vector<float> input_data_1;
    input_data_0.resize(1 * 50 * 9);
    input_data_1.resize(1 * 50 * 2);
    for (int i = 0; i < 1 * 50 * 9; i++) {
        input_data_0[i] = 1.f;
    }
    for (int i = 0; i < 1 * 50 * 2; i++) {
        input_data_1[i] = 1.f;
    }
    // 推理,获取输出
    const string model_path = "../model_test.onnx";
    OnnxInfer onnx_infer(model_path);

    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < 100; i++)
    {

        vector<float> output_data = onnx_infer.forward(input_data_0, input_data_1);
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

