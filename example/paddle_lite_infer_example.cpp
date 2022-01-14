#include <string.h>
#include <iostream>
#include "paddle_lite_infer.h"

int main(int argc, char const *argv[]) {
    std::string pl_yaml_path = "/data/huangshiqing/DeepLearning/code/MyNet/resource/conf/paddle_lite/paddle_lite_run.yaml";
    std::string input_yaml = "/data/huangshiqing/DeepLearning/code/MyNet/resource/input/feed.yaml";
    
    auto pl_infer = new PaddleLiteInfer();
    pl_infer->load_model(pl_yaml_path);
    pl_infer->init_inputs_outputs(input_yaml);

    pl_infer->init_infer_inputs_outputs();

    pl_infer->prepare_infer_inputs(input_yaml);
    
    pl_infer->prepare_inputs();
    pl_infer->infer_model();
    pl_infer->get_outputs();

    auto output_tensors = pl_infer->get_infer_outputs();
    
    //print
    for(auto& output_tensor : output_tensors) {
        std::cout<< *(float*)(output_tensor.second.data_) << std::endl;
    }
    
    delete pl_infer;
    return 0;
}