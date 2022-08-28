#include <stdint.h>

#include "type.h"

#ifdef __cplusplus
extern "C" {
#endif

void* wrapper_openpose_init(const char* model_file);
void wrapper_openpose_delete(void* p);
uint8_t* wrapper_openpose_load_image(void* p, const char* image_path, int dst_w, int dst_h, layout l);
void wrapper_openpose_nhwc2nchw(void* p, uint8_t* dst_data, uint8_t* src_data, int width, int height);
void wrapper_openpose_pre_process(void* p, uint8_t* data);
void wrapper_openpose_forward(void* p);
void wrapper_openpose_post_process(void* p);
void wrapper_openpose_get_outputs(void* p, PoseKeyPoint** pose_keypoints, int* count);

#ifdef __cplusplus
}
#endif