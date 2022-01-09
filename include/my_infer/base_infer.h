#pragma once

#include <map>
#include <string>

#include "tensor.h"

class BaseInfer {
private:
    /* data */
public:
    BaseInfer(/* args */) = default;
    ~BaseInfer() = default;

    virtual int load_model(std::string yaml_path) = 0;
    // virtual std::vector<std::string> get_input_names() = 0;
    // virtual std::vector<std::string> get_output_names() = 0;
    // virtual std::vector<std::string> get_middle_names() = 0;

    virtual void init_infer_inputs_outputs() = 0;
    void prepare_infer_inputs(void* input_data);//app->my_infer
    void prepare_infer_inputs(std::map<std::string, void*>& input_datas);//app->my_infer
    void prepare_infer_inputs(std::string file_path);//app->my_infer
    std::map<std::string, Tensor> get_infer_outputs();//my_infer->app

    virtual void prepare_inputs() = 0;//my_infer->mnn, paddle-lite...
    virtual void infer_model() = 0;
    virtual void get_outputs() = 0;//mnn, paddle-lite...->my_infer

    virtual std::map<std::string, Tensor> get_inputs() = 0;
    virtual std::map<std::string, Tensor> get_middles() = 0;

    std::map<std::string, Tensor> input_tensors_;
    std::map<std::string, Tensor> output_tensors_;
    std::map<std::string, Tensor> middle_tensors_;
};
