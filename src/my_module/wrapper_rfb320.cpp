#include "wrapper_rfb320.h"

#include "rfb320.h"

void* wrapper_rfb320_init(const char* model_file) {
    return new RFB320(model_file);
}

void wrapper_rfb320_delete(void* p) {
    delete static_cast<RFB320*>(p);
    return;
}

uint8_t* wrapper_rfb320_load_image(void* p, const char* image_path, int dst_w, int dst_h, layout l) {
    return static_cast<RFB320*>(p)->image.load_image(image_path, dst_w, dst_h, l);
}

void wrapper_rfb320_nhwc2nchw(void* p, uint8_t* dst_data, uint8_t* src_data, int width, int height) {
    static_cast<RFB320*>(p)->image.nhwc2nchw(dst_data, src_data, width, height);
    return;
}

void wrapper_rfb320_pre_process(void* p, uint8_t* data) {
    static_cast<RFB320*>(p)->pre_process(data);
    return;
}

void wrapper_rfb320_forward(void* p) {
    static_cast<RFB320*>(p)->forward();
    return;
}

void wrapper_rfb320_post_process(void* p) {
    static_cast<RFB320*>(p)->post_process();
    return;
}

void wrapper_rfb320_get_detections(void* p, Detection** detections, int* count) {
    *detections = static_cast<RFB320*>(p)->get_detections().data();
    *count = static_cast<RFB320*>(p)->get_detections().size();
    return;
}