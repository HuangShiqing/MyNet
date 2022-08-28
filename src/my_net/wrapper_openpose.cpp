#include "wrapper_openpose.h"

#include "openpose.h"

void* wrapper_openpose_init(const char* model_file) {
    return new OpenPose(model_file);
}

void wrapper_openpose_delete(void* p) {
    delete static_cast<OpenPose*>(p);
    return;
}

uint8_t* wrapper_openpose_load_image(void* p, const char* image_path, int dst_w, int dst_h, layout l) {
    return static_cast<OpenPose*>(p)->image.load_image(image_path, dst_w, dst_h, l);
}

void wrapper_openpose_nhwc2nchw(void* p, uint8_t* dst_data, uint8_t* src_data, int width, int height) {
    static_cast<OpenPose*>(p)->image.nhwc2nchw(dst_data, src_data, width, height);
    return;
}

void wrapper_openpose_pre_process(void* p, uint8_t* data) {
    static_cast<OpenPose*>(p)->pre_process(data);
    return;
}

void wrapper_openpose_forward(void* p) {
    static_cast<OpenPose*>(p)->forward();
    return;
}

void wrapper_openpose_post_process(void* p) {
    static_cast<OpenPose*>(p)->post_process();
    return;
}

void wrapper_openpose_get_outputs(void* p, PoseKeyPoint** pose_keypoints, int* count) {
    *pose_keypoints = static_cast<OpenPose*>(p)->get_outputs().data();
    *count = static_cast<OpenPose*>(p)->get_outputs().size();
    return;
}