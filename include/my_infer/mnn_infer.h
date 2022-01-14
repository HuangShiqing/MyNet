#pragma once
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "MNN/ImageProcess.hpp"
#include "base_infer.h"

class MNNInfer : public BaseInfer {
private:
    /* data */
public:
    MNNInfer(/* args */);
    ~MNNInfer();

    int load_model(std::string yaml_path);
    int init_inputs_outputs();
    // void init_inputs_outputs(void* input_data);//TODO:
    // void init_inputs_outputs(std::map<std::string, void*>& input_datas);//TODO:
    int init_inputs_outputs(std::string file_path);
        
    std::vector<std::string> get_input_names();
    std::vector<std::string> get_output_names();
    std::vector<std::string> get_middle_names();
    void init_infer_inputs_outputs();
    void prepare_inputs();//my_infer->mnn, paddle-lite...
    void infer_model();
    void get_outputs();//mnn, paddle-lite...->my_infer
    std::map<std::string, Tensor> get_inputs();
    std::map<std::string, Tensor> get_middles();

    MNN::Interpreter* interpreter_;
    MNN::Session* session_;
};
