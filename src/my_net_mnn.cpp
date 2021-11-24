#include "my_net_mnn.h"
#include "my_net.h"

void mynet_create_handle_mnn(void** handle, const int device_id) {
    handle_mnn* p = new handle_mnn;
    *handle = p;
}

void mynet_delete_handle_mnn(void* handle) {
    handle_mnn* p = (handle_mnn*)handle;
    mynet_releasemodel_mnn(handle);
    p->mnn_interpreter = nullptr;
    p->mnn_session = nullptr;
}

// TODO:maybe need set device
int mynet_init_from_file_mnn(void* handle, const char* model_file) {
    handle_mnn* p = (handle_mnn*)handle;
    p->mnn_interpreter = MNN::Interpreter::createFromFile(model_file);

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
    config.backendConfig = &backendConfig;
    p->mnn_session = p->mnn_interpreter->createSession(config);

    return 0;
}

int mynet_init_input_output_mnn(void* handle, std::map<const std::string, struct mynet_input>& inputs,
                                std::map<const std::string, struct mynet_feature>& outputs) {
    handle_mnn* p = (handle_mnn*)handle;

    auto mnn_inputs = p->mnn_interpreter->getSessionInputAll(p->mnn_session);
    for (auto& mnn_input : mnn_inputs) {
        mynet_input input;
        input.input_shape = mnn_input.second->shape();
        input.input_data = nullptr;
        int count = 1;
        for (auto& dim_size : input.input_shape) {
            count *= dim_size;
        }
        input.input_data = new uint8_t[count];
        inputs.insert(std::make_pair(mnn_input.first, input));
    }

    auto mnn_outputs = p->mnn_interpreter->getSessionOutputAll(p->mnn_session);
    for (auto& mnn_output : mnn_outputs) {
        mynet_feature output;
        output.output_shape = mnn_output.second->shape();
        output.output_data = mnn_output.second->host<float>();
        outputs.insert(std::make_pair(mnn_output.first, output));
    }
    return 0;
}

int mynet_prepare_forward_mnn(void* handle, std::map<const std::string, struct mynet_input>& inputs) {
    handle_mnn* p = (handle_mnn*)handle;

    std::map<std::string, MNN::Tensor*> input_tensors = p->mnn_interpreter->getSessionInputAll(p->mnn_session);
    for (auto& input_tensor : input_tensors) {
        auto input = inputs.find(input_tensor.first);
        if (input != inputs.end()) {
            auto nchwTensor = MNN::Tensor::create<float>(input->second.input_shape, (void*)input->second.input_data, MNN::Tensor::CAFFE);
            
            input_tensor.second->copyFromHostTensor(nchwTensor);
            // TODO:
            // input_tensor->host<float>() = (float*)input.second.input_data;
            // int count = 1;
            // for (auto& dim_size : input.second.input_shape) {
            //     count *= dim_size;
            // }
            // memcpy(input_tensor->host<float>(), (float*)input.second.input_data, count);
        }
    }
    return 0;
}

int mynet_forward_mnn(void* handle, std::map<const std::string, struct mynet_input>& inputs) {
    handle_mnn* p = (handle_mnn*)handle;

    p->mnn_interpreter->runSession(p->mnn_session);

    return 0;
}

int mynet_get_output_mnn(void* handle, std::map<const std::string, struct mynet_feature>& outputs) {
    handle_mnn* p = (handle_mnn*)handle;

    std::map<std::string, MNN::Tensor*> output_tensors = p->mnn_interpreter->getSessionOutputAll(p->mnn_session);
    for (auto& output_tensor : output_tensors) {
        auto output = outputs.find(output_tensor.first);
        if (output != outputs.end()) {
            struct mynet_feature fea;
            fea.output_shape = output_tensor.second->shape();
            fea.output_data = (void*)output_tensor.second->host<float>();
            output->second = fea;
        }
    }
    return 0;
}

int mynet_releasemodel_mnn(void* handle) {
    handle_mnn* p = (handle_mnn*)handle;
    p->mnn_interpreter->releaseModel();
    p->mnn_interpreter->releaseSession(p->mnn_session);

    return 0;
}