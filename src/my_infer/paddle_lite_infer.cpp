#include "paddle_lite_infer.h"

#include <memory.h>

#include <iostream>
#include <numeric>

#include "data_loader.h"
#include "yaml-cpp/yaml.h"

PaddleLiteInfer::PaddleLiteInfer(/* args */) {
}

PaddleLiteInfer::~PaddleLiteInfer() {
    // only for copy
    // for (auto& input_tensor : input_tensors_) {
    //     delete [] input_tensor.second.data_;
    // }
}

int PaddleLiteInfer::load_model(std::string yaml_path) {
    YAML::Node yaml_node = YAML::LoadFile(yaml_path);
    std::string model_dir = yaml_node["model_dir"].as<std::string>();
    std::vector<std::string> unset_passes = yaml_node["unset_passes"].as<std::vector<std::string>>();

    // 1.1 Set CxxConfig
    paddle::lite_api::CxxConfig config;
    config.set_model_dir(model_dir);
    config.set_threads(1);
    config.set_power_mode(paddle::lite_api::PowerMode::LITE_POWER_HIGH);
    config.set_valid_places({paddle::lite_api::Place{TARGET(kXPU), PRECISION(kFloat)}});
    config.set_xpu_workspace_l3_size_per_thread();
    config.set_xpu_multi_encoder_method("int16");
    config.unset_passes_internal(unset_passes);

    predictor_ = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::CxxConfig>(config);
}

// std::vector<std::string> PaddleLiteInfer::get_input_names() {
//     return predictor_->GetInputNames();
// };

// std::vector<std::string> PaddleLiteInfer::get_output_names() {
//     return predictor_->GetOutputNames();
// };

// std::vector<std::string> PaddleLiteInfer::get_middle_names() {
//     return predictor_->GetMiddleNames();
// };

int PaddleLiteInfer::init_inputs_outputs() {
    // TODO:
}

int PaddleLiteInfer::init_inputs_outputs(std::string file_path) {
    auto data_loader = DataLoader(file_path);

    std::vector<std::string> input_names = predictor_->GetInputNames();
    for (auto& yaml_tensor : data_loader.input_tensors_) {
        auto found = std::find(input_names.begin(), input_names.end(), yaml_tensor.first);
        if (found != input_names.end()) {
            std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor_->GetInputByName(*found)));

            paddle::lite_api::lod_t lod;
            for (auto lv0 : yaml_tensor.second.lod_) {
                std::vector<size_t> tmp;
                for (auto lv1 : lv0) {
                    tmp.push_back(static_cast<size_t>(lv1));
                }
                lod.push_back(tmp);
            }

            paddle::lite_api::shape_t shape;
            for (auto dim : yaml_tensor.second.shape_) {
                shape.push_back(static_cast<int>(dim));
            }
            input_tensor->Resize(shape);
            input_tensor->SetLoD(lod);

            int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            std::vector<float> data(count);
            memcpy(input_tensor->mutable_data<float>(), data.data(), count * sizeof(float));
        } else {
            std::cout << "[hsq] error" << std::endl;
            throw;
        }
    }

    predictor_->Run();  // InferShape
}

void PaddleLiteInfer::init_infer_inputs_outputs() {
    std::vector<std::string> input_names = predictor_->GetInputNames();
    for (auto input_name : input_names) {
        // auto input_tensor = predictor_->GetInputByName(input_name);//error
        std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor_->GetInputByName(input_name)));
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
        t.data_ = input_tensor->mutable_data<float>();  // link
        // t.data_ = new uint8_t[t.data_size_];  // copy
        // t.lod_ =
        input_tensors_.insert(std::make_pair(input_name, t));
    }

    std::vector<std::string> output_names = predictor_->GetOutputNames();
    for (size_t i = 0; i < output_names.size(); i++) {
        auto output_tensor = predictor_->GetOutput(i);
        auto shape = output_tensor->shape();
        int count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        if (count < 0)
            return;

        Tensor t;
        t.name_ = output_names[i];
        t.data_size_ = count * sizeof(float);
        t.data_ = output_tensor->mutable_data<float>();  // link
        // t.data_ = new uint8_t[t.data_size_];  // copy
        output_tensors_.insert(std::make_pair(output_names[i], t));
    }
}

void PaddleLiteInfer::prepare_inputs() {
    // only for copy
    // std::vector<std::string> input_names = predictor_->GetInputNames();
    // for (auto input_name : input_names) {
    //     // auto input_tensor = predictor_->GetInputByName(input_name);//error
    //     std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(predictor_->GetInputByName(input_name)));
    //     auto found = input_tensors_.find(input_name);
    //     if( found != input_tensors_.end()) {
    //         paddle::lite_api::lod_t lod;
    //         for (auto lv0 : found->second.lod_) {
    //             std::vector<size_t> tmp;
    //             for (auto lv1 : lv0) {
    //                 tmp.push_back(static_cast<size_t>(lv1));
    //             }
    //             lod.push_back(tmp);
    //         }
    //         input_tensor->SetLoD(lod);
    //         memcpy(input_tensor->mutable_data<float>(), found->second.data_, found->second.data_size_);
    //     } else {
    //         std::cout << "[hsq] error" << std::endl;
    //         throw;
    //     }
    // }
}

void PaddleLiteInfer::infer_model() {
    predictor_->Run();
};

void PaddleLiteInfer::get_outputs() {
    // only for copy
    // std::vector<std::string> pl_output_names = predictor_->GetOutputNames();
    // for (auto& output_tensor : output_tensors_) {
    //     auto found = std::find(pl_output_names.begin(), pl_output_names.end(), output_tensor.first);
    //     if (found != pl_output_names.end()) {
    //         auto pl_output_tensor = predictor_->GetOutput(found - pl_output_names.begin());

    //         // TODO:
    //         // output_tensor.second.lod_ = ;
    //         memcpy(output_tensor.second.data_, pl_output_tensor->mutable_data<float>(),
    //         output_tensor.second.data_size_);
    //     }
    // }
}

std::map<std::string, Tensor> PaddleLiteInfer::get_inputs(){
    // TODO:
};

std::map<std::string, Tensor> PaddleLiteInfer::get_middles(){
    // std::vector<std::string> middle_names = predictor_->GetMiddleNames();
    // for (int i = 0; i < middle_names.size(); i++) {
    //     std::unique_ptr<const paddle::lite_api::Tensor> middle_tensor =
    //         std::move(predictor_->GetMutableTensor(middle_names[i]));

    //     // TODO: tmp skip the other type
    //     // if(middle_tensor->type()!=paddle_infer::DataType::FLOAT32)
    //     //     continue;
    //     auto shape = middle_tensor->shape();
    //     int cout = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());

    //     Tensor middle_yaml;
    //     middle_yaml.name_ = middle_names[i];

    //     std::vector<int> tmp_shape;
    //     for (auto dim : shape) {
    //         tmp_shape.push_back(static_cast<int>(dim));
    //     }
    //     middle_yaml.shape_ = tmp_shape;

    //     std::vector<float> data(cout);
    //     middle_tensor->CopyToCpu<float>(data.data());
    //     middle_yaml.data_ = data;

    //     middle_yaml.lod_.clear();
    //     for (auto lv0 : middle_tensor->lod()) {
    //         std::vector<int> tmp;
    //         for (auto lv1 : lv0) {
    //             tmp.push_back(static_cast<int>(lv1));
    //         }
    //         middle_yaml.lod_.push_back(tmp);
    //     }

    //     middle_tensors_.insert(std::make_pair(middle_names[i], middle_yaml));
    // }
    // return middle_tensors_;  // TODO:
};