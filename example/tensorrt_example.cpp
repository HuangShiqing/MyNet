#include <string.h>
#include <iostream>
#include "tensorrt_infer.h"

int main(int argc, char const *argv[]) {
    std::string trt_yaml_path = "/home/hsq/DeepLearning/code/MyNet/resource/conf/tensorrt/tensorrt.yaml";
    std::string input_yaml = "/home/hsq/DeepLearning/code/MyNet/resource/input/mnist.yaml";
    
    auto trt_infer = new TensorRTInfer();
    trt_infer->load_model(trt_yaml_path);
    trt_infer->init_inputs_outputs();

    trt_infer->init_infer_inputs_outputs();

    trt_infer->prepare_infer_inputs(input_yaml);
    
    trt_infer->prepare_inputs();
    trt_infer->infer_model();
    trt_infer->get_outputs();

    auto output_tensors = trt_infer->get_infer_outputs();
    
    //softmax
    for(auto& output_tensor : output_tensors) {
        auto shape = output_tensor.second.shape_;
        int confidence_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        std::vector<float> confidence(confidence_size);

        float sum=0;
        for(int i=0;i<confidence_size;i++) {
            float data = *((float*)(output_tensor.second.data_)+i);
            confidence[i] = exp(data);
            sum+=confidence[i];
        }
        for(int i=0;i<confidence_size;i++) {
            confidence[i] /= sum;
        }
        float max = -1;
        int max_index = -1;
        for(int i=0;i<confidence_size;i++) {
            if(confidence[i]>max) {
                max = confidence[i];
                max_index = i;
            }
        }

        std::cout<<"max_index: "<<max_index<<std::endl;
        std::cout<<"max_confidence: "<<max<<std::endl;
    }

    delete trt_infer;
    return 0;
}