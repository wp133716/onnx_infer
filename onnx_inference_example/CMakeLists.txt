cmake_minimum_required(VERSION 3.13)

project(onnxruntime_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# link_directories(
#     /home/user/3rd-party/onnx/lib
# )

add_executable(${PROJECT_NAME} onnx_test.cpp)
target_link_libraries(${PROJECT_NAME} onnxruntime onnxruntime_providers_shared)

