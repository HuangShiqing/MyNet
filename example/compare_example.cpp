#include "compare.h"
#include <string.h>
#include <iostream>
#include "paddle_lite_infer.h"
#include "paddle_infer.h"

int main(int argc, char const *argv[])
{
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

    auto pl_output_tensors = pl_infer->get_infer_outputs();
    auto pl_middle_tensors = pl_infer->get_infer_middles();

// ------------------------------
    std::string pp_yaml_path = "/data/huangshiqing/DeepLearning/code/MyNet/resource/conf/paddle/paddle_run.yaml";
    // std::string input_yaml = "/data/huangshiqing/DeepLearning/code/MyNet/resource/input/feed.yaml";
    
    auto pp_infer = new PaddleInfer();
    pp_infer->load_model(pp_yaml_path);
    pp_infer->init_inputs_outputs(input_yaml);

    pp_infer->init_infer_inputs_outputs();

    pp_infer->prepare_infer_inputs(input_yaml);
    
    pp_infer->prepare_inputs();
    pp_infer->infer_model();
    pp_infer->get_outputs();

    auto pp_output_tensors = pp_infer->get_infer_outputs();   
    auto pp_middle_tensors = pp_infer->get_infer_middles();


// ------------------------------
    Compare compare;
    compare.compare_all(pl_infer, pp_infer, 0.1);
    compare.dump_tensor(pl_infer, pp_infer, "./result");
    compare.dump_result("./result/result.json");
    delete pp_infer;
    delete pl_infer;
    return 0;
}
