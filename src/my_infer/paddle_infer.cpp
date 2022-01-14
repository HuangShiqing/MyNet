#include "paddle_infer.h"

#include <memory.h>

#include <iostream>
#include <numeric>

#include "data_loader.h"

PaddleInfer::PaddleInfer(/* args */) {
}

PaddleInfer::~PaddleInfer() {
    for (auto& input_tensor : input_tensors_) {
        delete [] input_tensor.second.data_;
    }
}

int PaddleInfer::load_model(std::string yaml_path) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    std::string model_path = yaml_node["model_path"].as<std::string>();
    std::string param_path = yaml_node["param_path"].as<std::string>();
    
    // 1.1 Set CxxConfig
    paddle_infer::Config paddle_config;
    // paddle_config.SetModel(model_dir);//TODO:
    paddle_config.SetModel(model_path, param_path);
    // paddle_config.EnableUseGpu(100, 0);
    paddle_config.DisableGpu();
    paddle_config.EnableMKLDNN();

    predictor_ = paddle_infer::CreatePredictor(paddle_config);
}

// std::vector<std::string> PaddleInfer::get_input_names() {
//     return predictor_->GetInputNames();
// };

// std::vector<std::string> PaddleInfer::get_output_names() {
//     return predictor_->GetOutputNames();
// };

// std::vector<std::string> PaddleInfer::get_middle_names() {
//     return predictor_->GetMiddleNames();
// };

int PaddleInfer::init_inputs_outputs() {
    // TODO:
}

int PaddleInfer::init_inputs_outputs(std::string file_path) {
    auto data_loader = DataLoader(file_path);

    std::vector<std::string> input_names = predictor_->GetInputNames();
    for (auto& yaml_tensor : data_loader.input_tensors_) {
        auto found = std::find(input_names.begin(), input_names.end(), yaml_tensor.first);
        if (found != input_names.end()) {
            auto input_tensor(std::move(predictor_->GetInputHandle(*found)));  // Paddle
            
            std::vector<std::vector<size_t>> lod;
            for (auto lv0 : yaml_tensor.second.lod_) {
                std::vector<size_t> tmp;
                for (auto lv1 : lv0) {
                    tmp.push_back(static_cast<size_t>(lv1));
                }
                lod.push_back(tmp);
            }

            std::vector<int> shape;
            for (auto dim : yaml_tensor.second.shape_) {
                shape.push_back(static_cast<int>(dim));
            }

            input_tensor->Reshape(shape);
            input_tensor->SetLoD(lod);

            int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            std::vector<float> data(count);
            input_tensor->CopyFromCpu(data.data());
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }

    predictor_->Run();  // InferShape
}

void PaddleInfer::init_infer_inputs_outputs() {
    std::vector<std::string> input_names = predictor_->GetInputNames();
    for (auto input_name : input_names) {
        auto input_tensor(std::move(predictor_->GetInputHandle(input_name)));  // Paddle            
        auto shape = input_tensor->shape();

        int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (count < 0)
            return;

        std::vector<int> tmp_shape;
        for (auto dim : shape) {
            tmp_shape.push_back(static_cast<int>(dim));
        }

        Tensor t;
        t.name_ = input_name;
        // t.dtype_ =
        t.shape_ = tmp_shape;
        t.data_size_ = count * sizeof(float);
        t.data_ = new uint8_t[t.data_size_];  // copy

        // t.lod_ =
        input_tensors_.insert(std::make_pair(input_name, t));
    }

    std::vector<std::string> output_names = predictor_->GetOutputNames();
    for (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = predictor_->GetOutputHandle(output_names[i]);
        auto shape = output_tensor->shape();
        int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (count < 0)
            return;

        auto lod = output_tensor->lod();

        Tensor t;
        t.name_ = output_names[i];
        t.shape_ = shape;
        t.data_size_ = count * sizeof(float);
        t.data_ = new uint8_t[t.data_size_];  // copy
        output_tensors_.insert(std::make_pair(output_names[i], t));
    }
}

void PaddleInfer::prepare_inputs() {
    std::vector<std::string> input_names = predictor_->GetInputNames();
    for (auto input_name : input_names) {
        std::unique_ptr<paddle_infer::Tensor> input_tensor(
            std::move(predictor_->GetInputHandle(input_name)));  // Paddle
        auto found = input_tensors_.find(input_name);
        if( found != input_tensors_.end()) {

            std::vector<std::vector<size_t>> lod;
            for (auto lv0 : found->second.lod_) {
                std::vector<size_t> tmp;
                for (auto lv1 : lv0) {
                    tmp.push_back(static_cast<size_t>(lv1));
                }
                lod.push_back(tmp);
            }
            input_tensor->SetLoD(lod);
            input_tensor->CopyFromCpu<float>((float*)found->second.data_);
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }
}

// void PaddleInfer::prepare_inputs(std::string file_path) {
//     auto data_loader = DataLoader(file_path);

//     std::vector<std::string> input_names = predictor_->GetInputNames();
//     for (auto input_name : input_names) {
//         std::unique_ptr<paddle_infer::Tensor> input_tensor(
//             std::move(predictor_->GetInputHandle(input_name)));  // Paddle

//         auto input_yaml = data_loader.input_tensors_.find(input_name);
//         if (input_yaml != data_loader.input_tensors_.end()) {
//             std::vector<std::vector<size_t>> lod;
//             for (auto lv0 : input_yaml->second.lod_) {
//                 std::vector<size_t> tmp;
//                 for (auto lv1 : lv0) {
//                     tmp.push_back(static_cast<size_t>(lv1));
//                 }
//                 lod.push_back(tmp);
//             }

//             std::vector<int> shape;
//             for (auto dim : input_yaml->second.shape_) {
//                 shape.push_back(static_cast<int>(dim));
//             }
//             input_tensor->SetLoD(lod);
//             input_tensor->Reshape(shape);
//             input_tensor->CopyFromCpu(input_yaml->second.data_.data());
//         } else {
//             std::cout << "[hsq] miss " << input_name << " in the yaml_file";
//             throw;
//         }
//     }
//     // // 1.3.2 or random init input_tensor
//     // else {
//     //     auto shape = input_tensor->shape();
//     //     shape[0] = 1;
//     //     int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
//     //     input_tensor->Resize(shape);

//     //     std::vector<float> input_data;
//     //     input_data.resize(cout);
//     //     std::generate(input_data.begin(), input_data.end(), []() { return rand() / static_cast<float>(RAND_MAX); });
//     //     memcpy(input_tensor->mutable_data<float>(), input_data.data(), cout * sizeof(float));

//     //     // TODO: lod
//     //     // input_tensor->SetLoD(input_yaml->second.lod);
//     // }
// };

void PaddleInfer::infer_model() {
    CHECK(predictor_->Run());
};

void PaddleInfer::get_outputs() {
    std::vector<std::string> output_names = predictor_->GetOutputNames();
    for (int i = 0; i < output_names.size(); i++) {
        auto output_tensor = predictor_->GetOutputHandle(output_names[i]);
        auto found = output_tensors_.find(output_names[i]);
        if (found != output_tensors_.end()){
            output_tensor->CopyToCpu((float*)found->second.data_);
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }
}

std::map<std::string, Tensor> PaddleInfer::get_inputs(){
    // TODO:
};

// std::map<std::string, Tensor> PaddleInfer::get_outputs() {
//     std::vector<std::string> output_names = predictor_->GetOutputNames();
//     for (int i = 0; i < output_names.size(); i++) {
//         auto output_tensor = predictor_->GetOutputHandle(output_names[i]);
//         auto shape = output_tensor->shape();
//         int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

//         Tensor output_yaml;
//         output_yaml.name_ = output_names[i];

//         std::vector<int> tmp_shape;
//         for (auto dim : shape) {
//             tmp_shape.push_back(static_cast<int>(dim));
//         }
//         output_yaml.shape_ = tmp_shape;

//         std::vector<float> data(cout);
//         output_tensor->CopyToCpu(data.data());
//         output_yaml.data_ = data;

//         output_yaml.lod_.clear();
//         for (auto lv0 : output_tensor->lod()) {
//             std::vector<int> tmp;
//             for (auto lv1 : lv0) {
//                 tmp.push_back(static_cast<int>(lv1));
//             }
//             output_yaml.lod_.push_back(tmp);
//         }

//         output_tensors_.insert(std::make_pair(output_names[i], output_yaml));
//     }
//     return output_tensors_;  // TODO:
// };

std::map<std::string, Tensor> PaddleInfer::get_middles() {
//     std::vector<std::string> middle_names = predictor_->GetMiddleNames();
//     for (int i = 0; i < middle_names.size(); i++) {
//         auto middle_tensor = predictor_->GetOutputHandle(middle_names[i]);

//         // TODO: tmp skip the other type
//         // if(middle_tensor->type()!=paddle_infer::DataType::FLOAT32)
//         //     continue;
//         auto shape = middle_tensor->shape();
//         int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

//         Tensor middle_yaml;
//         middle_yaml.name_ = middle_names[i];

//         std::vector<int> tmp_shape;
//         for (auto dim : shape) {
//             tmp_shape.push_back(static_cast<int>(dim));
//         }
//         middle_yaml.shape_ = tmp_shape;

//         std::vector<float> data(cout);
//         middle_tensor->CopyToCpu<float>(data.data());
//         middle_yaml.data_ = data;

//         middle_yaml.lod_.clear();
//         for (auto lv0 : middle_tensor->lod()) {
//             std::vector<int> tmp;
//             for (auto lv1 : lv0) {
//                 tmp.push_back(static_cast<int>(lv1));
//             }
//             middle_yaml.lod_.push_back(tmp);
//         }

//         middle_tensors_.insert(std::make_pair(middle_names[i], middle_yaml));
//     }
//     return middle_tensors_;  // TODO:
};