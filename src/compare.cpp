#include "compare.h"

#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
Compare::Compare(/* args */) {
}

Compare::~Compare() {
}

void Compare::compare(std::map<std::string, Tensor> targets, std::map<std::string, Tensor> refs, float percent,
                      tensor_type type) {
    std::vector<std::string> not_found;
    std::vector<std::string> error;
    std::vector<std::string> correct;
    for (auto& target : targets) {
        auto found = refs.find(target.first);
        if (found == refs.end()) {
            if (type == tensor_type::output) {
                found = refs.find(target.first.substr(0, target.first.size() - 13));  // Paddle-Lite erase /target_trans
                if (found == refs.end()) {
                    // std::cout << "[hsq] " << target.first << " not found in refs. " << std::endl;
                    not_found.push_back(target.first);
                    continue;
                } else {
                    ;
                }
            } else {
                // std::cout << "[hsq] " << target.first << " not found in refs. " << std::endl;
                not_found.push_back(target.first);
                continue;
            }
        }

        if (found->second.shape_ != target.second.shape_) {
            std::cout << "[hsq] " << target.first << " shape not equal" << std::endl;
            error.push_back(target.first);
            continue;
        }

        int error_index = -1;
        for (int i = 0; i < target.second.data_size_ / sizeof(float); i++) {
            float a = *((float*)(target.second.data_) + i);
            float b = *((float*)(found->second.data_) + i);
            if (fabs((a - b) / b) > percent) {
                error_index = i;
                break;
            }
        }
        if (error_index == -1) {
            // std::cout << "[hsq] " << target.first << " data equal" << std::endl;
            correct.push_back(target.first);
        } else {
            std::cout << "[hsq] " << target.first << " and " << found->second.name_ << " data Not equal from index "
                      << error_index << std::endl;
            error.push_back(target.first);
            for (int i = error_index; i < error_index + 10; i++) {
                float a = *((float*)(target.second.data_) + i);
                float b = *((float*)(found->second.data_) + i);
                std::cout << "Target: " << a << std::endl;
                std::cout << "Ref: " << b << std::endl;
            }
            // TODO:dump
        }
    }
    switch (type) {
        case tensor_type::output:
            result_outputs_.clear();
            result_outputs_.insert(std::make_pair("not_found", not_found));
            result_outputs_.insert(std::make_pair("error", error));
            result_outputs_.insert(std::make_pair("correct", correct));
            break;
        case tensor_type::middle:
            result_middles_.clear();
            result_middles_.insert(std::make_pair("not_found", not_found));
            result_middles_.insert(std::make_pair("error", error));
            result_middles_.insert(std::make_pair("correct", correct));
            break;
        case tensor_type::input:
            result_inputs_.clear();
            result_inputs_.insert(std::make_pair("not_found", not_found));
            result_inputs_.insert(std::make_pair("error", error));
            result_inputs_.insert(std::make_pair("correct", correct));
            break;
        default:
            break;
    }
}

void Compare::compare_all(BaseInfer* target, BaseInfer* ref, float percent) {
    compare(target->output_tensors_, ref->output_tensors_, percent, tensor_type::output);
    compare(target->middle_tensors_, ref->middle_tensors_, percent, tensor_type::middle);
    // compare(target->input_tensors_, ref->input_tensors_, tensor_type::input);
}

void Compare::dump_result(std::string json_path) {
    std::ofstream result_file;
    result_file.open(json_path.c_str(), std::ios::out);

    result_file << "{" << std::endl;

    std::vector<std::string> errors;
    for (auto& error : result_outputs_["error"]) {
        errors.push_back(error);
    }
    for (auto& error : result_middles_["error"]) {
        errors.push_back(error);
    }
    for (auto& error : result_inputs_["error"]) {
        errors.push_back(error);
    }
    result_file << "    \"error\": [";
    for (int i = 0; i < errors.size(); i++) {
        result_file << "\"" << errors[i] << "\"";
        if (i != errors.size() - 1)
            result_file << ", ";
    }
    result_file << "]," << std::endl;

    std::vector<std::string> corrects;
    for (auto& correct : result_outputs_["correct"]) {
        corrects.push_back(correct);
    }
    for (auto& correct : result_middles_["correct"]) {
        corrects.push_back(correct);
    }
    for (auto& correct : result_inputs_["correct"]) {
        corrects.push_back(correct);
    }
    result_file << "    \"correct\": [";
    for (int i = 0; i < corrects.size(); i++) {
        result_file << "\"" << corrects[i] << "\"";
        if (i != corrects.size() - 1)
            result_file << ", ";
    }
    result_file << "]" << std::endl;

    result_file << "}" << std::endl;
    result_file.close();
}

void Compare::dump_tensor(BaseInfer* target, BaseInfer* ref, std::string dir) {
    int ret;
    std::string cmd_rm = "rm -rf " + dir;
    ret = system(cmd_rm.c_str());
    std::string cmd_error_dir = "mkdir -p " + dir + "/error/";
    std::string cmd_correct_dir = "mkdir -p " + dir + "/correct/";
    ret = system(cmd_error_dir.c_str());
    ret = system(cmd_correct_dir.c_str());

    std::vector<std::string> errors;
    for (auto& error : result_outputs_["error"]) {
        errors.push_back(error);
    }
    for (auto& error : result_middles_["error"]) {
        errors.push_back(error);
    }
    for (auto& error : result_inputs_["error"]) {
        errors.push_back(error);
    }
    std::vector<std::string> corrects;
    for (auto& correct : result_outputs_["correct"]) {
        corrects.push_back(correct);
    }
    for (auto& correct : result_middles_["correct"]) {
        corrects.push_back(correct);
    }
    for (auto& correct : result_inputs_["correct"]) {
        corrects.push_back(correct);
    }

    for (auto& tensor_name : corrects) {
        std::ofstream result_file;
        result_file.open(dir + "/correct/" + tensor_name + "#0" + ".txt", std::ios::out);
        if (target->output_tensors_.find(tensor_name) != target->output_tensors_.end() ||
            target->middle_tensors_.find(tensor_name) != target->middle_tensors_.end()) {
            std::map<std::string, Tensor>::iterator found;
            if (target->output_tensors_.find(tensor_name) != target->output_tensors_.end()) {
                found = target->output_tensors_.find(tensor_name);
            } else {
                found = target->middle_tensors_.find(tensor_name);
            }
            for (size_t i = 0; i < found->second.data_size_ / sizeof(float); i++) {
                float a = *((float*)(found->second.data_) + i);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(10) << a;
                result_file << stream.str() << " ";
            }
            result_file << std::endl;
        }
        result_file.close();

        result_file.open(dir + "/correct/" + tensor_name + "#1" + ".txt", std::ios::out);
        if (ref->output_tensors_.find(tensor_name) != ref->output_tensors_.end() ||
            ref->middle_tensors_.find(tensor_name) != ref->middle_tensors_.end()) {
            std::map<std::string, Tensor>::iterator found;
            if (ref->output_tensors_.find(tensor_name) != ref->output_tensors_.end()) {
                found = ref->output_tensors_.find(tensor_name);
            } else {
                found = ref->middle_tensors_.find(tensor_name);
            }
            for (size_t i = 0; i < found->second.data_size_ / sizeof(float); i++) {
                float a = *((float*)(found->second.data_) + i);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(6) << a;
                result_file << stream.str() << " ";
            }
            result_file << std::endl;
        }
        result_file.close();
    }

    for (auto& tensor_name : errors) {
        std::ofstream result_file;
        result_file.open(dir + "/error/" + tensor_name + ".txt", std::ios::out);
        if (target->output_tensors_.find(tensor_name) != target->output_tensors_.end() ||
            target->middle_tensors_.find(tensor_name) != target->middle_tensors_.end()) {
            std::map<std::string, Tensor>::iterator found;
            if (target->output_tensors_.find(tensor_name) != target->output_tensors_.end()) {
                found = target->output_tensors_.find(tensor_name);
            } else {
                found = target->middle_tensors_.find(tensor_name);
            }
            for (size_t i = 0; i < found->second.data_size_ / sizeof(float); i++) {
                float a = *((float*)(found->second.data_) + i);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(6) << a;
                result_file << stream.str() << " ";
            }
            result_file << std::endl;
        }
        result_file.close();

        result_file.open(dir + "/error/" + tensor_name + "#ref" + ".txt", std::ios::out);
        if (ref->output_tensors_.find(tensor_name) != ref->output_tensors_.end() ||
            ref->middle_tensors_.find(tensor_name) != ref->middle_tensors_.end()) {
            std::map<std::string, Tensor>::iterator found;
            if (ref->output_tensors_.find(tensor_name) != ref->output_tensors_.end()) {
                found = ref->output_tensors_.find(tensor_name);
            } else {
                found = ref->middle_tensors_.find(tensor_name);
            }
            for (size_t i = 0; i < found->second.data_size_ / sizeof(float); i++) {
                float a = *((float*)(found->second.data_) + i);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(6) << a;
                result_file << stream.str() << " ";
            }
            result_file << std::endl;
        }
        result_file.close();
    }
}