#include "mnn_infer.h"

#include <memory.h>

#include <iostream>
#include <map>
#include <numeric>

#include "data_loader.h"
#include "yaml-cpp/yaml.h"


MNNInfer::MNNInfer(/* args */) {
};

MNNInfer::~MNNInfer() {
    for (auto& input_tensor : input_tensors_) {
        delete [] input_tensor.second.data_;
    }
};

int MNNInfer::load_model(std::string yaml_path) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    std::string model_path = yaml_node["model_path"].as<std::string>();
    // TODO:
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
    config.backendConfig = &backendConfig;

    interpreter_ = MNN::Interpreter::createFromFile(model_path.c_str());
    session_ = interpreter_->createSession(config);
}

void MNNInfer::init_infer_inputs_outputs() {
    auto mnn_inputs = interpreter_->getSessionInputAll(session_);
    for (auto& mnn_input : mnn_inputs) {
        auto shape = mnn_input.second->shape();
        // TODO: input shape may include -1 or ?
        int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

        Tensor t;
        t.name_ = mnn_input.first;
        // t.dtype_ =
        t.shape_ = shape;
        t.data_size_ = cout * sizeof(float);
        t.data_ = new uint8_t[cout * sizeof(float)];  // copy
        // t.lod_ =
        input_tensors_.insert(std::make_pair(mnn_input.first, t));
    }

    auto mnn_outputs = interpreter_->getSessionOutputAll(session_);
    for (auto& mnn_output : mnn_outputs) {
        auto shape = mnn_output.second->shape();
        int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

        Tensor t;
        t.name_ = mnn_output.first;
        // t.dtype_ =
        t.shape_ = shape;
        t.data_size_ = cout * sizeof(float);
        t.data_ = mnn_output.second->host<float>();  // link
        // t.lod_ =
        output_tensors_.insert(std::make_pair(mnn_output.first, t));
    }
}

void MNNInfer::prepare_inputs() {
    auto mnn_inputs = interpreter_->getSessionInputAll(session_);
    for (auto& input_tensor : input_tensors_) {
        auto found = mnn_inputs.find(input_tensor.first);
        if (found != mnn_inputs.end()) {
            auto nchwTensor =
                MNN::Tensor::create<float>(input_tensor.second.shape_, input_tensor.second.data_, MNN::Tensor::CAFFE);
            found->second->copyFromHostTensor(nchwTensor);
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }
}

void MNNInfer::infer_model() {
    interpreter_->runSession(session_);
}

void MNNInfer::get_outputs() {
    auto mnn_outputs = interpreter_->getSessionOutputAll(session_);
    for (auto& mnn_output : mnn_outputs) {
        auto found = output_tensors_.find(mnn_output.first);
        if (found != output_tensors_.end()) {
            auto shape = mnn_output.second->shape();
            int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

            found->second.name_ = mnn_output.first;
            // found->second.dtype_
            found->second.shape_ = shape;
            found->second.data_size_ = cout * sizeof(float);
            // found->second.data_ //link
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }
}

std::map<std::string, Tensor> MNNInfer::get_inputs(){
    // TODO:
};

std::map<std::string, Tensor> MNNInfer::get_middles() {
    // TODO:
}