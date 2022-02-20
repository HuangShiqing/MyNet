#define once

#include "base_infer.h"
#include "yaml-cpp/yaml.h"

// nv
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
// common
#include "buffers.h"

class TensorRTInfer: public BaseInfer {
private:
    /* data */
public:
    TensorRTInfer(/* args */);
    ~TensorRTInfer();

    int load_model(std::string yaml_path);
    int init_inputs_outputs();
    // void init_inputs_outputs(void* input_data);//TODO:
    // void init_inputs_outputs(std::map<std::string, void*>& input_datas);//TODO:
    int init_inputs_outputs(std::string file_path);

    // std::vector<std::string> get_input_names();
    // std::vector<std::string> get_output_names();
    // std::vector<std::string> get_middle_names();

    void init_infer_inputs_outputs();

    // prepare and get fun only for copy mode or link mode's user copy work
    void prepare_inputs();  // my_infer->mnn, paddle-lite...
    void infer_model();
    void get_outputs();  // mnn, paddle-lite...->my_infer

    void get_inputs();
    void get_middles();

    nvinfer1::INetworkDefinition* network_;
    nvinfer1::IExecutionContext* context_;
    samplesCommon::BufferManager* buffers_;
};


