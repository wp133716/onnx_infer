# onnx_infer
Use onnxruntime to deploy the model in the c++ for inference

 ### tensorflow训练好的模型使用ONNX Runtime在C++部署
 #### 环境
 - ubuntu20.04
 - cuda 11.6
 - cudnn 8.2.4 
#### 参考链接
- [https://blog.csdn.net/mightbxg/article/details/119237326](https://blog.csdn.net/mightbxg/article/details/119237326)
## ONNX Runtime
ONNX (Open Neural Network Exchange) 是微软和脸书主导的深度学习开发工具生态系统，ONNX Runtime (简称 ORT) 则是微软开发的跨平台高性能机器学习训练与推理加速器，根据官方的说法推理/训练速度最高能有 17X/1.4X 的提升，其优异的性能非常适合深度学习模型部署。

#### 克隆
```bash
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime/
git checkout v1.13.0
```
ONNXRuntime版本和cuda、cudnn版本要对应，具体参考[官方链接:https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
这里选择了1.13.0版本
ONNX Runtime      | CUDA  |  cuDNN
--------------------       | ----------- |------
1.14 /1.13.1 / 1.13                |  11.6	|8.2.4 (Linux) / 8.5.0.96 (Windows)
#### 编译
```bash
./build.sh --skip_tests --use_cuda --config Release --build_shared_lib --parallel --cuda_home /usr/local/cuda-11.6 --cudnn_home /usr/local/cuda-11.6
```
--use_cuda表示build with CUDA support，cuda_home和cudnn_home指向cuda和cudnn的安装路径
- 注意
编译过程中会链接其它github仓库(大概几十个)，可能因为网络问题导致编译失败，需要科学上网或者手动添加镜像源
```bash
cd ${your git repo root}
cd .git
vim config
```
修改为：
```bash
[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
    ignorecase = true
    precomposeunicode = true
[remote "origin"]
    url = https://github.com.cnpmjs.org/microsoft/onnxruntime.git
    fetch = +refs/tags/v1.13.0:refs/tags/v1.13.0
```
#### 编译完成，安装
```bash
cd ./build/Linux/release
make install DESTDIR=想要安装的路径
```
#### 配置环境变量
```bash
# onnxruntime
export ONNX_HOME=/home/user/3rd-party/onnx/usr/local
export PATH=$PATH:$ONNX_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ONNX_HOME/lib
export LIBRARY_PATH=$LIBRARY_PATH:$ONNX_HOME/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$ONNX_HOME/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$ONNX_HOME/include
```