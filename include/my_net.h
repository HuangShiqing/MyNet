#pragma once

#include <stddef.h>
#include <map>
#include <vector>

enum class FrameworkType {
    MNN = 0,
};
enum class DeviceType {
    CPU = 0,
};
struct mynet_feature {
    std::vector<int> output_shape;  
    void* output_data;              
    mynet_feature() : output_data(NULL) {
    }
};
struct mynet_input {
    std::vector<int> input_shape;
    void* input_data;
    mynet_input() : input_data(NULL) {
    }
};

class MyNet {
public:
    MyNet(FrameworkType framework_type = FrameworkType::MNN, DeviceType device_type = DeviceType::CPU,
          const int device_id = 0);
    ~MyNet(void);

    int InitModelFile(const char* model_file);
    
    int InitInputsOutputs();

    int PrepareForward();//拷贝填充输入data或者更新输入data_ptr
    
    int Forward();
    
    int GetOutput();
    
    int ReleaseModel();

    std::map<const std::string, mynet_input> m_inputs;
    std::map<const std::string, mynet_feature> m_outputs;    
protected:
    void* pHandle;
};