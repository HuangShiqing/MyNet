#include <stdint.h>

#include "type.h"

#ifdef __cplusplus
extern "C" {
#endif

void* wrapper_rfb320_init(const char* model_file);
void wrapper_rfb320_delete(void* p);
uint8_t* wrapper_rfb320_load_image(void* p, const char* image_path, int dst_w, int dst_h, layout l);
void wrapper_rfb320_nhwc2nchw(void* p, uint8_t* dst_data, uint8_t* src_data, int width, int height);
void wrapper_rfb320_pre_process(void* p, uint8_t* data);
void wrapper_rfb320_forward(void* p);
void wrapper_rfb320_post_process(void* p);
void wrapper_rfb320_get_detections(void* p, Detection** detections, int* count);

#ifdef __cplusplus
}
#endif