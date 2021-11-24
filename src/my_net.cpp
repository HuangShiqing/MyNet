#include <my_net.h>
#include <my_net_mnn.h>

MyNet::MyNet(FrameworkType framework_type, DeviceType device_type, const int device_id) {
    switch (framework_type) {
        case FrameworkType::MNN:
            mynet_create_handle_mnn(&pHandle, device_id);
            break;
        default:
            break;
    }
}

MyNet::~MyNet(void) {
    if (pHandle != NULL) {
        mynet_delete_handle_mnn(pHandle);
    }
    pHandle = NULL;
}

int MyNet::InitModelFile(const char* model_file) {
    return mynet_init_from_file_mnn(pHandle, model_file);
}

int MyNet::InitInputsOutputs() {
    return mynet_init_input_output_mnn(pHandle, m_inputs, m_outputs);
}

int MyNet::PrepareForward() {
    return mynet_prepare_forward_mnn(pHandle, m_inputs);
}

int MyNet::Forward() {  
    return mynet_forward_mnn(pHandle, m_inputs);
}

int MyNet::GetOutput() {
    return mynet_get_output_mnn(pHandle, m_outputs);
}

int MyNet::ReleaseModel() {
    return mynet_releasemodel_mnn(pHandle);
}
