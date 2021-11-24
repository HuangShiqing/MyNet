#pragma once
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "MNN/ImageProcess.hpp"


typedef struct {
    MNN::Interpreter* mnn_interpreter;
    MNN::Session* mnn_session;    
} handle_mnn;

void mynet_create_handle_mnn(void** handle, const int device_id);

void mynet_delete_handle_mnn(void* handle);

int mynet_init_from_file_mnn(void* handle, const char* model_file);

int mynet_init_input_output_mnn(void* handle, std::map<const std::string, struct mynet_input>& inputs, std::map<const std::string, struct mynet_feature>& outputs);

int mynet_prepare_forward_mnn(void* handle, std::map<const std::string, struct mynet_input>& inputs);

int mynet_forward_mnn(void* handle, std::map<const std::string, struct mynet_input>& inputs);

int mynet_get_output_mnn(void* handle, std::map<const std::string, struct mynet_feature>& outputs);

int mynet_releasemodel_mnn(void* handle);