#include "base_infer.h"

#include <memory.h>

#include <iostream>
#include <map>
#include <string>
#include <numeric>

#include "data_loader.h"

void BaseInfer::prepare_infer_inputs(void* input_data) {
    // TODO:
}

void BaseInfer::prepare_infer_inputs(std::map<std::string, void*>& input_datas) {
    for (auto& input_data : input_datas) {
        auto found = input_tensors_.find(input_data.first);
        if (found != input_tensors_.end()) {
            memcpy(found->second.data_, input_data.second, found->second.data_size_);
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }
}

void BaseInfer::prepare_infer_inputs(std::string file_path) {
    auto data_loader = DataLoader(file_path);

    for (auto& yaml_tensor : data_loader.input_tensors_) {
        auto found = input_tensors_.find(yaml_tensor.first);
        if (found != input_tensors_.end()) {
            // TODO:check
            auto shape = yaml_tensor.second.shape_;
            int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

            found->second.shape_ = yaml_tensor.second.shape_;
            found->second.lod_ = yaml_tensor.second.lod_;
            found->second.dtype_ = yaml_tensor.second.dtype_;
            found->second.data_size_ = count * sizeof(float);
            memcpy(found->second.data_, yaml_tensor.second.data_, found->second.data_size_);
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }
    // // 1.3.2 or random init input_tensor
    // else {
    //     auto shape = input_tensor->shape();
    //     shape[0] = 1;
    //     int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    //     input_tensor->Resize(shape);

    //     std::vector<float> input_data;
    //     input_data.resize(cout);
    //     std::generate(input_data.begin(), input_data.end(), []() { return rand() / static_cast<float>(RAND_MAX); });
    //     memcpy(input_tensor->mutable_data<float>(), input_data.data(), cout * sizeof(float));

    //     // TODO: lod
    //     // input_tensor->SetLoD(input_yaml->second.lod);
    // }    
};


std::map<std::string, Tensor> BaseInfer::get_infer_outputs() {
    return output_tensors_;
}