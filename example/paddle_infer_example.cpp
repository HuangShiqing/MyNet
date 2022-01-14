#include <string.h>
#include <iostream>
#include "paddle_infer.h"

int main(int argc, char const *argv[]) {
    std::string pp_yaml_path = "/data/huangshiqing/DeepLearning/code/MyNet/resource/conf/paddle/paddle_run.yaml";
    std::string input_yaml = "/data/huangshiqing/DeepLearning/code/MyNet/resource/input/feed.yaml";
    
    auto pp_infer = new PaddleInfer();
    pp_infer->load_model(pp_yaml_path);
    pp_infer->init_inputs_outputs(input_yaml);

    pp_infer->init_infer_inputs_outputs();

    pp_infer->prepare_infer_inputs(input_yaml);
    
    pp_infer->prepare_inputs();
    pp_infer->infer_model();
    pp_infer->get_outputs();

    auto output_tensors = pp_infer->get_infer_outputs();
    
    //print
    for(auto& output_tensor : output_tensors) {
        std::cout<< *(float*)(output_tensor.second.data_) << std::endl;
    }
    
    delete pp_infer;
    return 0;
}