#include <string.h>
#include <iostream>
#include "mnn_infer.h"


int main(int argc, char const *argv[])
{
    std::string mnn_yaml_path = "/home/hsq/DeepLearning/code/MyNet/resource/conf/mnn/rfb320.yaml";
    
    auto mnn_infer = new MNNInfer();
    mnn_infer->load_model(mnn_yaml_path);
    mnn_infer->init_inputs_outputs();

    mnn_infer->init_infer_inputs_outputs();

    // mnn_infer->prepare_infer_inputs();
    
    mnn_infer->prepare_inputs();
    mnn_infer->infer_model();
    mnn_infer->get_outputs();

    auto output_tensors = mnn_infer->get_infer_outputs();
    
    //print
    for(auto& output_tensor : output_tensors) {
        std::cout<< *(float*)(output_tensor.second.data_) << std::endl;
    }
    
    delete mnn_infer;
    return 0;
}
