cmake_minimum_required(VERSION 3.10)
project(onnx_infer)

set(CMAKE_CXX_STANDARD 11)

# debug
set(CMAKE_BUILD_TYPE "Debug")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# onnxruntime
# include_directories(/usr/local/include/onnxruntime)
# link_directories(/usr/local/lib)

add_executable(onnx_infer
                main.cpp onnx_infer.h onnx_infer.cpp)

target_link_libraries(onnx_infer ${OpenCV_LIBS} onnxruntime)